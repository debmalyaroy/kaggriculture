"""Sequential probability ratio test for accepting a candidate agent.

Why this exists
---------------
This project keeps being fooled by small match counts. The worked example is in
the notes: `cost_per_animal_day` 4.5 -> 3.5 measured 80% over 10 matches and
56% over 16. A fixed-n test with n=10 has no power to detect the size of edge we
are actually hunting, so it alternates between rubber-stamping noise and
rejecting real gains.

SPRT fixes both ends. It is the test used by computer-chess engine testing
(fishtest) for exactly this problem: cheap noisy paired games, tiny true edges,
and a need to stop as soon as the answer is known rather than after a fixed
budget. You state two hypotheses in Elo,

    H0: the candidate is no better than the incumbent   (elo0, default 0)
    H1: the candidate is better by at least elo1        (elo1, default 25)

and it accumulates a log-likelihood ratio after every game, stopping the moment
the evidence crosses either bound.

Choose elo1 for your compute budget, not for your ambition. The cost of a test
is set by how small an edge you insist on resolving, and it is brutally
non-linear -- run `python -m kaggriculture.measure.sprt` for the measured table. At ~7s a match
and 8 workers, elo1=25 is roughly an hour for a decisive answer on a marginal
change and minutes on an obvious one; elo1=8 is chess-engine territory and wants
tens of thousands of games. The default is 25 because a change smaller than that
will not move a leaderboard placing anyway.

"inconclusive" at the cap is a real answer, not a failure: it means the
difference is smaller than elo1, so keep the incumbent.

Usage
-----
    from kaggriculture.measure.sprt import SPRT
    t = SPRT(elo0=0, elo1=25, alpha=0.05, beta=0.05)
    for ...:
        t.record(win=True)              # or win=False, or draw=True
        if t.verdict() != "continue":
            break
    print(t.summary())

Draws are recorded but carry no information under the pentanomial-free
simplification used here; in this game exact ties are vanishingly rare.
"""
import math

__all__ = ["SPRT", "elo_to_p", "p_to_elo", "elo_interval"]


def elo_to_p(elo):
    """Expected score for a player `elo` points stronger."""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def p_to_elo(p):
    """Elo difference implied by a win probability."""
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -400.0 * math.log10(1.0 / p - 1.0)


def elo_interval(wins, games, conf=0.95):
    """(low, mid, high) Elo for `wins` out of `games`, normal approximation.

    Reported alongside every verdict because a bare "accepted" hides how wide
    the estimate still is.
    """
    if games <= 0:
        return (float("-inf"), 0.0, float("inf"))
    p = wins / games
    z = 1.959963985 if conf >= 0.95 else 1.644853627
    se = math.sqrt(max(p * (1 - p), 1e-9) / games)
    return (p_to_elo(max(p - z * se, 1e-6)),
            p_to_elo(p),
            p_to_elo(min(p + z * se, 1 - 1e-6)))


class SPRT:
    """Wald's sequential test over paired match results.

    Parameters
    ----------
    elo0, elo1 : the null and alternative Elo gains. elo1 is the smallest
        improvement worth adopting -- set it to the gain you would actually
        spend a submission slot on, not to the smallest gain that exists.
    alpha, beta : false-accept and false-reject rates.
    max_games : hard cap. Reaching it returns "inconclusive", which means
        "smaller than elo1, keep the incumbent", not "unknown".
    """

    def __init__(self, elo0=0.0, elo1=25.0, alpha=0.05, beta=0.05,
                 max_games=400):
        if elo1 <= elo0:
            raise ValueError("elo1 must exceed elo0")
        self.elo0, self.elo1 = float(elo0), float(elo1)
        self.p0, self.p1 = elo_to_p(elo0), elo_to_p(elo1)
        self.lower = math.log(beta / (1.0 - alpha))      # reject H1
        self.upper = math.log((1.0 - beta) / alpha)      # accept H1
        self.max_games = int(max_games)
        self.llr = 0.0
        self.wins = self.losses = self.draws = 0

    # ----------------------------------------------------------- recording --
    @property
    def games(self):
        return self.wins + self.losses + self.draws

    def record(self, win=None, draw=False):
        """One game. `win=True` candidate won, `win=False` incumbent won."""
        if draw:
            self.draws += 1
            return self.verdict()
        if win:
            self.wins += 1
            self.llr += math.log(self.p1 / self.p0)
        else:
            self.losses += 1
            self.llr += math.log((1.0 - self.p1) / (1.0 - self.p0))
        return self.verdict()

    def record_many(self, wins, games):
        for i in range(games):
            self.record(win=(i < wins))
        return self.verdict()

    # ------------------------------------------------------------ decision --
    def verdict(self):
        if self.llr >= self.upper:
            return "accept"
        if self.llr <= self.lower:
            return "reject"
        if self.games >= self.max_games:
            return "inconclusive"
        return "continue"

    def decided(self):
        return self.verdict() != "continue"

    def progress(self):
        """How far through the test we are, 0..1, for a progress line."""
        if self.llr >= 0:
            return min(1.0, self.llr / self.upper) if self.upper else 0.0
        return min(1.0, self.llr / self.lower) if self.lower else 0.0

    def summary(self):
        decided = self.wins + self.losses
        rate = (self.wins / decided) if decided else 0.0
        lo, mid, hi = elo_interval(self.wins, decided or 1)
        v = self.verdict()
        note = {
            "accept": f"candidate is better by at least {self.elo1:.0f} Elo",
            "reject": f"candidate is not better by {self.elo1:.0f} Elo -- keep the incumbent",
            "inconclusive": (f"no decision in {self.games} games: the difference "
                             f"is smaller than {self.elo1:.0f} Elo, so it is not "
                             f"worth a submission slot"),
            "continue": "still running",
        }[v]
        return (f"{v.upper():<12} {self.wins}W-{self.losses}L-{self.draws}D "
                f"({rate:.1%})  Elo {mid:+.0f} [{lo:+.0f}, {hi:+.0f}]  "
                f"llr {self.llr:+.2f} of [{self.lower:.2f}, {self.upper:.2f}]\n"
                f"             {note}")


def _demo(elo1=25.0, max_games=400, trials=200):
    """What each kind of candidate actually costs, measured not asserted.

    Median games to a decision, and how often the test gets it right, over
    `trials` simulated runs per true edge.
    """
    import random
    import statistics
    print(f"SPRT  elo0=0  elo1={elo1:.0f}  alpha=beta=0.05  cap={max_games}")
    print(f"(a match is ~7s; 8 workers -> ~1.1s of wall clock per game)\n")
    print(f"{'true edge':>10}  {'median games':>12}  {'wall clock':>10}  verdicts")
    for true_elo in (-100, -40, -15, 0, 15, 25, 50, 100, 240):
        p = elo_to_p(true_elo)
        counts, games = {"accept": 0, "reject": 0, "inconclusive": 0}, []
        for k in range(trials):
            rng = random.Random(9000 + int(true_elo) * 97 + k)
            t = SPRT(elo0=0, elo1=elo1, max_games=max_games)
            while not t.decided():
                t.record(win=(rng.random() < p))
            counts[t.verdict()] += 1
            games.append(t.games)
        med = statistics.median(games)
        mins = med * 1.1 / 60.0
        breakdown = "  ".join(f"{k}={v * 100 // trials}%"
                              for k, v in counts.items() if v)
        print(f"{true_elo:>+9} Elo  {med:>12.0f}  {mins:>8.0f}m   {breakdown}")
    print("\nRead the +15 row before trusting any 10-match result: an edge that "
          "size\nis not reliably detectable at all inside a few hundred games.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--elo1", type=float, default=25.0)
    ap.add_argument("--max-games", type=int, default=400)
    ap.add_argument("--trials", type=int, default=200)
    a = ap.parse_args()
    _demo(a.elo1, a.max_games, a.trials)
