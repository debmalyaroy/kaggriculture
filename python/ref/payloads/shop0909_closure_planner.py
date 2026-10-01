# decoded from agents/v61.1_bandit.py line 1017; exec namespace: _PLANNER_NS
"""E182 modification: Shop0909 last-seven-turn physical closure planner.

No engine imports, policy tapes, replay fixtures, RNG or remote calls.
The only supported market continuation is SELL; unknown execution abstains.
"""
from copy import deepcopy
from time import perf_counter
START, FINAL = (712, 718)
OPS = set(FARMER_MOVES) | {'PASS', 'DROP', 'PICKUP', 'PLACE', 'PLANT', 'WATER', 'HARVEST', 'FERTILIZE', 'DIG', 'BUILD_COOP', 'BUILD_PASTURE', 'FEED', 'CARE', 'COLLECT_FERTILIZER'}
ITEMS = tuple(PRODUCTS) + tuple(ANIMALS)

class Unsupported(ValueError):
    pass

def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

def _settings(config):
    size, turns, last = (_get(config, k, d) for k, d in [('boardSize', 10), ('turnsPerDay', 24), ('episodeSteps', 720)])
    if (size, turns, last) != (10, 24, 720):
        raise Unsupported('requires pinned 10x10/24/720 terminal window')
    cap = int(_get(config, 'shedCapacity', 100))
    orders = min(10, int(_get(config, 'maxMarketOrdersPerTurn', 10)))
    if cap < 1 or orders < 1:
        raise Unsupported('invalid capacity/order limit')
    return (size, turns, cap, orders)

def physical_state(obs):
    """Comparable own physical state; market prices and bank are intentionally excluded."""
    seat = int(_get(obs, 'player', 0))
    farm = _get(obs, 'farms')[seat]
    return ({k: v for k, v in farm.items() if k != 'money'}, _get(obs, 'private'))

def _commands(action, n):
    return [action.get('farmer', ['PASS']), *action.get('hands', [])][:n] + [['PASS'] for _ in range(max(0, n - 1 - len(action.get('hands', []))))]

def _clone_state(farm, private):
    f = dict(farm)
    f['tiles'] = [[dict(tile) if isinstance(tile, dict) else tile for tile in row] for row in farm['tiles']]
    f['farmer'] = list(farm['farmer'])
    f['hands'] = [list(pos) for pos in farm['hands']]
    f['unlocked_quadrants'] = list(farm['unlocked_quadrants'])
    pr = dict(private)
    pr['shed'], pr['seeds'] = (dict(private['shed']), dict(private['seeds']))
    pr['inventories'] = [dict(inv) for inv in private['inventories']]
    return (f, pr)

def _clone_schedule(schedule):
    result = []
    for action in schedule:
        value = dict(action)
        if 'farmer' in action:
            value['farmer'] = list(action['farmer'])
        for key in ['hands', 'market']:
            if key in action:
                value[key] = [list(command) for command in action[key]]
        result.append(value)
    return result

def _validate(schedule, n, orders):
    for action in schedule:
        if not isinstance(action, dict) or set(action) - {'farmer', 'hands', 'market'}:
            raise Unsupported('unknown action shape')
        if not isinstance(action.get('hands', []), list):
            raise Unsupported('hands must be a list')
        for command in [action.get('farmer', ['PASS']), *action.get('hands', [])]:
            if not isinstance(command, list) or not command or command[0] not in OPS:
                raise Unsupported('unknown/malformed unit operation')
            if command[0] in {'PICKUP', 'PLACE', 'PLANT'}:
                if len(command) < 2 or command[1] not in ITEMS:
                    raise Unsupported('unknown unit item')
                if len(command) > 2 and (not isinstance(command[2], int)):
                    raise Unsupported('noninteger unit quantity')
        market = action.get('market', [])
        if not isinstance(market, list) or len(market) > orders:
            raise Unsupported('market order shape/cap')
        for order in market:
            if not isinstance(order, list) or len(order) != 3 or order[0] != 'SELL' or (order[1] not in PRODUCTS) or (not isinstance(order[2], int)) or (order[2] <= 0):
                raise Unsupported('baseline market must contain positive integer SELL only')

def liquidation(shed, inherited_market, max_orders=10):
    """Use actual post-unit stock; retain first parent item ordering, then stable product order."""
    items = []
    for order in inherited_market:
        if order[1] not in items:
            items.append(order[1])
    items += [item for item in PRODUCTS if item not in items]
    orders = [['SELL', item, int(shed.get(item, 0))] for item in items if shed.get(item, 0) > 0]
    if len(orders) > min(10, max_orders):
        raise Unsupported('actual final stock exceeds order slots')
    return orders

def shop_liquidation(farm, private, prices):
    """Exact original final worker/drop and market rule, with current stock/prices."""
    return liquidate(FarmView({'player': 0, 'farms': [farm], 'private': private, 'market': {'prices': prices}}))

def simulate(obs, config, schedule, *, final_liquidate=False, detailed=False, preserve_final_commands=False):
    """Exact own unit/decay and SELL-stock transitions. No claim to simulate shared prices."""
    size, turns, cap, order_cap = _settings(config)
    step = int(_get(obs, 'step', -1))
    if step < START or step + len(schedule) - 1 > FINAL or (not schedule):
        raise Unsupported('outside 712..718; no day boundary or terminal auto-drop')
    if any(((t + 1) % turns == 0 for t in range(step, step + len(schedule)))):
        raise Unsupported('day boundary')
    farm0, private0 = physical_state(obs)
    farm, private = _clone_state(farm0, private0)
    n = 1 + len(farm['hands'])
    if len(private['inventories']) != n or n > 32:
        raise Unsupported('invalid/unbounded worker inventory shape')
    _validate(schedule, n, order_cap)
    deposited = [dict() for _ in range(n)]
    sold = {}
    snapshots, rows, events = ([], [], [])
    executed = _clone_schedule(schedule)
    overflow = 0
    for offset, action in enumerate(executed):
        t = step + offset
        if detailed:
            snapshots.append(_clone_state(farm, private))
        if t == FINAL and (not preserve_final_commands):
            action = shop_liquidation(farm, private, _get(obs, 'market')['prices'])
            executed[offset] = action
        all_commands = [action.get('farmer', ['PASS']), *action.get('hands', [])]
        demand = {}
        for command in all_commands:
            if command[0] == 'PLANT':
                demand[command[1]] = demand.get(command[1], 0) + 1
        blocked = {item for item, count in demand.items() if count > private['seeds'].get(item, 0)}
        for actor, command in enumerate(_commands(action, n)):
            if command[0] == 'PLANT' and command[1] in blocked:
                command = ['PASS']
            pos = farm['farmer'] if actor == 0 else farm['hands'][actor - 1]
            xy = tuple(pos)
            inv = private['inventories'][actor]
            before_inv = dict(inv) if command[0] in {'DROP', 'HARVEST', 'COLLECT_FERTILIZER'} else None
            before_shed = dict(private['shed']) if command[0] in {'DROP', 'PLACE'} else None
            _apply_unit_action(farm, private, actor, command, size, t // turns, turns, cap)
            if before_shed is not None:
                delta = {item: amount - before_shed.get(item, 0) for item, amount in private['shed'].items() if amount > before_shed.get(item, 0)}
                for item, amount in delta.items():
                    deposited[actor][item] = deposited[actor].get(item, 0) + amount
                if delta:
                    events.append({'offset': offset, 'actor': actor, 'op': command[0], 'xy': xy, 'deposited': delta})
                if command[0] == 'DROP':
                    overflow += sum((max(0, amount - inv.get(item, 0) - delta.get(item, 0)) for item, amount in before_inv.items()))
            if command[0] in {'HARVEST', 'COLLECT_FERTILIZER'}:
                delta = {item: amount - before_inv.get(item, 0) for item, amount in inv.items() if amount > before_inv.get(item, 0)}
                if delta:
                    events.append({'offset': offset, 'actor': actor, 'op': command[0], 'xy': xy, 'acquired': delta})
        pre_market = dict(private['shed'])
        if t == FINAL and preserve_final_commands and final_liquidate:
            action['market'] = liquidation(pre_market, [], order_cap)
            prices = _get(obs, 'market')['prices']
            action['market'].sort(key=lambda order: -int(prices.get(order[1], 0)) * order[2])
        for _, item, requested in action.get('market', []):
            quantity = min(requested, private['shed'].get(item, 0), 99999)
            if quantity > 0:
                private['shed'][item] -= quantity
                sold[item] = sold.get(item, 0) + quantity
        _decay_plants(farm, t)
        rows.append({'pre_market_shed': pre_market, 'post_market_shed': dict(private['shed']), 'deposited_by_actor': [dict(v) for v in deposited], 'sold': dict(sold)})
    if detailed:
        snapshots.append(_clone_state(farm, private))
    return {'rows': rows, 'states': snapshots, 'events': events, 'actions': executed, 'overflow_units': overflow, 'farm': farm, 'private': private, 'sold': sold}

def _ge(left, right):
    return all((left.get(item, 0) >= value for item, value in right.items()))

def dominates(candidate, baseline):
    """Preserve every baseline worker's actual deposit prefixes and shed availability."""
    if candidate['overflow_units']:
        return False
    for new, old in zip(candidate['rows'], baseline['rows']):
        if not _ge(new['pre_market_shed'], old['pre_market_shed']):
            return False
        if not _ge(new['sold'], old['sold']):
            return False
        if any((not _ge(a, b) for a, b in zip(new['deposited_by_actor'], old['deposited_by_actor']))):
            return False
    return True

def _value(run, prices):
    shed = run['private']['shed']
    return sum(((run['sold'].get(item, 0) + shed.get(item, 0)) * prices[item] for item in PRODUCTS))

def _walk(start, end):
    x, y = start
    tx, ty = end
    return [['EAST']] * max(0, tx - x) + [['WEST']] * max(0, x - tx) + [['SOUTH']] * max(0, ty - y) + [['NORTH']] * max(0, y - ty)

def _return(pos):
    targets = _shed_access_tiles(10)
    target = min(targets, key=lambda xy: (abs(pos[0] - xy[0]) + abs(pos[1] - xy[1]), targets.index(xy)))
    return _walk(pos, target) + [['DROP']]

def _proposals(run, actor, prices, max_per_actor):
    """One/two resource bundles plus direct carry closure, replacing a baseline suffix."""
    owners = {}
    for event in run['events']:
        if 'acquired' in event:
            owners.setdefault((tuple(event['xy']), event['op']), set()).add(event['actor'])
    proposals = []
    seen = set()
    horizon = len(run['rows'])
    for offset in range(horizon):
        farm, private = run['states'][offset]
        pos = tuple(farm['farmer'] if actor == 0 else farm['hands'][actor - 1])
        inventory = private['inventories'][actor]
        carried = sum((prices.get(item, 0) * count for item, count in inventory.items()))
        prefix_deposits = run['rows'][offset - 1]['deposited_by_actor'][actor] if offset else {}
        future_deposits = run['rows'][-1]['deposited_by_actor'][actor]
        obligation = sum((prices.get(item, 0) * (count - prefix_deposits.get(item, 0)) for item, count in future_deposits.items()))
        bundles = []
        for y, row in enumerate(farm['tiles']):
            for x, tile in enumerate(row):
                if not isinstance(tile, dict):
                    continue
                xy, operations, value = ((x, y), [], 0)
                if tile.get('yield_units', 0) > 0:
                    item = tile.get('crop') if tile.get('kind') == 'PLANT' else ANIMALS.get(tile.get('animal'), {}).get('product')
                    mature = item and ('animal' in tile or (START + offset) // 24 - tile['planted_day'] >= CROPS[item]['first_yield_day'])
                    if mature and (not owners.get((xy, 'HARVEST'), set()) - {actor}):
                        operations.append(['HARVEST'])
                        value += prices[item] * tile['yield_units']
                if tile.get('fertilizer_available') and 'animal' in tile and (not owners.get((xy, 'COLLECT_FERTILIZER'), set()) - {actor}):
                    operations.append(['COLLECT_FERTILIZER'])
                    value += prices['FERTILIZER']
                if operations:
                    distance = len(_walk(pos, xy)) + len(operations) + len(_return(xy))
                    if distance <= horizon - offset:
                        bundles.append((xy, operations, value, distance))
        bundles.sort(key=lambda b: (-b[2] / b[3], -b[2], b[0]))
        variants = [([], carried)] if carried else []
        for xy, ops, value, _ in bundles[:6]:
            variants.append(([(xy, ops)], carried + value))
        for first in bundles[:3]:
            for second in bundles[:3]:
                if first[0] != second[0]:
                    variants.append(([(first[0], first[1]), (second[0], second[1])], carried + first[2] + second[2]))
        for stops, value in variants:
            route, cursor = ([], pos)
            for xy, ops in stops:
                route += _walk(cursor, xy) + ops
                cursor = xy
            route += _return(cursor)
            if len(route) > horizon - offset:
                continue
            route += [['PASS']] * (horizon - offset - len(route))
            key = (offset, tuple((tuple(c) for c in route)))
            if key not in seen:
                seen.add(key)
                proposals.append((value - obligation, offset, route, len(stops)))
    proposals.sort(key=lambda p: (-p[0], p[1], p[2]))
    direct = [p for p in proposals if p[3] == 0 and p[0] > 0][:2]
    chosen = direct + [p for p in proposals if p not in direct]
    return chosen[:max_per_actor]

def plan_terminal(obs, config, baseline_remaining, *, max_simulations=64, passes=1, proposals_per_actor=4):
    """At 712 accept seven actions; positive physical delivery is mandatory."""
    begun = perf_counter()
    fallback = {'accepted': False, 'reason': '', 'actions': None, 'simulations': 0}
    try:
        if int(_get(obs, 'step', -1)) != START or len(baseline_remaining) != FINAL - START + 1:
            raise Unsupported('planning requires step 712 and exactly seven actions through 718')
        max_simulations = min(256, max(1, int(max_simulations)))
        passes = min(2, max(1, int(passes)))
        proposals_per_actor = min(16, max(1, int(proposals_per_actor)))
        baseline = simulate(obs, config, baseline_remaining, detailed=True)
        prices = {item: max(1, float(_get(obs, 'market', {}).get('prices', {}).get(item, 1))) for item in PRODUCTS}
        current, best = (_clone_schedule(baseline_remaining), baseline)
        baseline_value = best_value = _value(baseline, prices)
        changes, simulations = ([], 0)
        n = len(baseline['private']['inventories'])
        for sweep in range(passes):
            improved = False
            for actor in range(n):
                winner = None
                for _, offset, route, bundle_count in _proposals(best, actor, prices, proposals_per_actor):
                    if simulations >= max_simulations:
                        break
                    trial = _clone_schedule(current)
                    for i, command in enumerate(route, offset):
                        if actor == 0:
                            trial[i]['farmer'] = command
                        else:
                            trial[i].setdefault('hands', [])
                            while len(trial[i]['hands']) < n - 1:
                                trial[i]['hands'].append(['PASS'])
                            trial[i]['hands'][actor - 1] = command
                    evaluated = simulate(obs, config, trial)
                    simulations += 1
                    score = _value(evaluated, prices)
                    if score > best_value and dominates(evaluated, baseline):
                        required = {(tuple(e['xy']), e['op'], e['actor']): e['acquired'] for e in best['events'] if 'acquired' in e and e['actor'] != actor}
                        acquired = {}
                        for e in evaluated['events']:
                            if 'acquired' in e:
                                key = (tuple(e['xy']), e['op'], e['actor'])
                                dst = acquired.setdefault(key, {})
                                for item, amount in e['acquired'].items():
                                    dst[item] = dst.get(item, 0) + amount
                        if all((_ge(acquired.get(k, {}), v) for k, v in required.items())):
                            winner, best_value = ((trial, offset, bundle_count), score)
                if winner:
                    current, offset, bundle_count = winner
                    best = simulate(obs, config, current, detailed=True)
                    changes.append({'pass': sweep, 'actor': actor, 'from_step': START + offset, 'resource_bundles': bundle_count, 'estimated_stock_value': best_value})
                    improved = True
                if simulations >= max_simulations:
                    break
            if not improved or simulations >= max_simulations:
                break
        if not changes or best_value <= baseline_value:
            return {**fallback, 'reason': 'no positive physical delivery gain', 'simulations': simulations, 'changed_workers': [], 'changes': [], 'certificate': {'stock_value_gain_at_initial_prices': 0, 'sold_unit_delta': dict.fromkeys(PRODUCTS, 0)}, 'planning_ms': (perf_counter() - begun) * 1000}
        final = simulate(obs, config, current, final_liquidate=True, detailed=True)
        physical = simulate(obs, config, current)
        if not dominates(physical, baseline):
            raise Unsupported('no zero-overflow dominating continuation')
        delta = {item: final['sold'].get(item, 0) - baseline['sold'].get(item, 0) for item in PRODUCTS}
        deposited_gain = any((final['rows'][-1]['deposited_by_actor'][actor].get(item, 0) > baseline['rows'][-1]['deposited_by_actor'][actor].get(item, 0) for actor in range(n) for item in PRODUCTS))
        worker_change = any((_commands(new, n) != _commands(old, n) for new, old in zip(final['actions'], baseline['actions'])))
        accepted = worker_change and deposited_gain and any((v > 0 for v in delta.values())) and all((v >= 0 for v in delta.values()))
        plan = {'accepted': accepted, 'reason': 'joint physical dominance' if accepted else 'no improvement', 'baseline': _clone_schedule(baseline_remaining), 'actions': final['actions'], 'expected_states': final['states'][:-1], 'simulations': simulations, 'changes': changes, 'abandoned': False, 'changed_workers': sorted({c['actor'] for c in changes}), 'certificate': {'baseline_rows': baseline['rows'], 'physical_rows': physical['rows'], 'baseline_overflow': baseline['overflow_units'], 'candidate_overflow': final['overflow_units'], 'sold_unit_delta': delta, 'stock_value_gain_at_initial_prices': best_value - baseline_value, 'baseline_final_shed': baseline['private']['shed'], 'final_shed': final['private']['shed'], 'positive_physical_deposit_gain': deposited_gain, 'markets_712_717_unchanged': all((final['actions'][i].get('market', []) == baseline_remaining[i].get('market', []) for i in range(FINAL - START)))}}
    except (Unsupported, KeyError, TypeError, ValueError, IndexError) as exc:
        plan = {**fallback, 'reason': str(exc)}
    plan['planning_ms'] = (perf_counter() - begun) * 1000
    return plan

def _effective_action(action, n):
    return (_commands(action, n), action.get('market', []))

def _recover_observed(obs, config, parent_action, plan):
    """Bounded cargo salvage after deviation; never resume old positional commands."""
    farm, private = physical_state(obs)
    positions = [farm['farmer'], *farm['hands']]
    remaining = FINAL - int(_get(obs, 'step')) + 1
    room = max(0, int(_get(config, 'shedCapacity', 100)) - sum(private['shed'].values()))
    commands = []
    problems = []
    prices = _get(obs, 'market', {}).get('prices', {})
    for actor, (pos, inv) in enumerate(zip(positions, private['inventories'])):
        command = ['PASS']
        if any((v > 0 for v in inv.values())):
            route = _return(pos)
            if len(route) > remaining:
                problems.append({'actor': actor, 'reason': 'unreachable cargo'})
            elif len(route) > 1:
                command = route[0]
            elif sum((max(0, q) for q in inv.values())) <= room:
                command = ['DROP']
                room -= sum((max(0, q) for q in inv.values()))
            else:
                items = [item for item in PRODUCTS if inv.get(item, 0) > 0]
                if room and items:
                    item = max(items, key=lambda i: (prices.get(i, 1) * min(inv[i], room), -PRODUCTS.index(i)))
                    quantity = min(inv[item], room)
                    command = ['PLACE', item, quantity]
                    room -= quantity
                else:
                    problems.append({'actor': actor, 'reason': 'no shed capacity'})
        commands.append(command)
    action = {'farmer': commands[0], 'hands': commands[1:], 'market': deepcopy(parent_action.get('market', []))}
    if int(_get(obs, 'step')) == FINAL:
        action['market'] = []
        action = simulate(obs, config, [action], final_liquidate=True, preserve_final_commands=True)['actions'][0]
    plan['recovery_steps'] = plan.get('recovery_steps', 0) + 1
    if problems:
        plan.setdefault('recovery_failures', []).append({'step': int(_get(obs, 'step')), 'problems': problems})
    return action

def terminal_action(obs, config, parent_action, plan):
    """Canonical guard, pre-deviation abstention, observed recovery after deviation."""
    step = int(_get(obs, 'step', -1))
    if not plan or not plan.get('accepted') or (not START <= step <= FINAL):
        return parent_action
    if plan.get('abandoned'):
        return _recover_observed(obs, config, parent_action, plan) if plan.get('deviated') else parent_action
    index = step - START
    n = 1 + len(physical_state(obs)[0]['hands'])
    mismatch = physical_state(obs) != plan['expected_states'][index] or _effective_action(parent_action, n) != _effective_action(plan['baseline'][index], n)
    if mismatch:
        plan['abandoned'] = True
        plan['abandon_step'] = step
        plan['reason'] = 'physical observation or effective baseline action diverged'
        plan['safety_failure'] = True
        return _recover_observed(obs, config, parent_action, plan) if plan.get('deviated') else parent_action
    result = deepcopy(plan['actions'][index])
    if step == FINAL:
        farm, private = physical_state(obs)
        result = shop_liquidation(farm, private, _get(obs, 'market')['prices'])
    if _commands(result, n) != _commands(parent_action, n):
        plan['deviated'] = True
    return result