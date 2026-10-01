//! A self-describing row: every value carries its column name and one-line doc, so the
//! Parquet schema and `schema.json` are derived from the code that computes the values.

#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    I(Option<i64>),
    F(Option<f64>),
    S(Option<String>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Col {
    pub name: String,
    pub doc: &'static str,
    pub val: Val,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Row {
    pub cols: Vec<Col>,
}

impl Row {
    pub fn with_capacity(n: usize) -> Self {
        Row { cols: Vec::with_capacity(n) }
    }

    pub fn i(&mut self, name: impl Into<String>, doc: &'static str, v: Option<i64>) {
        self.cols.push(Col { name: name.into(), doc, val: Val::I(v) });
    }

    pub fn f(&mut self, name: impl Into<String>, doc: &'static str, v: Option<f64>) {
        self.cols.push(Col { name: name.into(), doc, val: Val::F(v) });
    }

    pub fn s(&mut self, name: impl Into<String>, doc: &'static str, v: Option<String>) {
        self.cols.push(Col { name: name.into(), doc, val: Val::S(v) });
    }

    pub fn get(&self, name: &str) -> Option<&Val> {
        self.cols.iter().find(|c| c.name == name).map(|c| &c.val)
    }

    /// Prepend identity columns.
    pub fn prepend(&mut self, mut head: Row) {
        head.cols.append(&mut self.cols);
        self.cols = head.cols;
    }
}
