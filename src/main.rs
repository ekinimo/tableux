use std::{
    collections::{
        hash_map::Entry::{Occupied, Vacant},
        HashMap, HashSet,
    },
    ops::{Index, IndexMut},
    vec,
    io::{self, BufRead},

};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct VarIdx(usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct FormulaIdx(usize);

#[derive(Copy, Clone, Debug)]
enum Formula {
    Var(VarIdx),
    Not(FormulaIdx),
    And(FormulaIdx, FormulaIdx),
    Or(FormulaIdx, FormulaIdx),
    Implies(FormulaIdx, FormulaIdx),
}

impl Index<VarIdx> for FormulaPool {
    type Output = String;
    fn index(&self, VarIdx(idx): VarIdx) -> &Self::Output {
        &self.var_names[idx]
    }
}
impl IndexMut<VarIdx> for FormulaPool {
    fn index_mut(&mut self, VarIdx(idx): VarIdx) -> &mut Self::Output {
        &mut self.var_names[idx]
    }
}

impl Index<FormulaIdx> for FormulaPool {
    type Output = Formula;
    fn index(&self, FormulaIdx(idx): FormulaIdx) -> &Self::Output {
        &self.formulas[idx]
    }
}
impl IndexMut<FormulaIdx> for FormulaPool {
    fn index_mut(&mut self, FormulaIdx(idx): FormulaIdx) -> &mut Self::Output {
        &mut self.formulas[idx]
    }
}

#[derive(Clone, Debug, Default)]
struct FormulaPool {
    formulas: Vec<Formula>,
    vars: HashMap<String, VarIdx>,
    var_names: Vec<String>,
}

impl FormulaPool {
    fn display(&self, idx: FormulaIdx) -> String {
        match self[idx] {
            Formula::Var(x) => self[x].to_owned(),
            Formula::Not(x) => format!("( ¬ {} )", self.display(x)),
            Formula::And(x, y) => format!("( {} ∧ {} )", self.display(x), self.display(y)),
            Formula::Or(x, y) => format!("( {} ∨ {} )", self.display(x), self.display(y)),
            Formula::Implies(x, y) => format!("( {} ⇒ {} )", self.display(x), self.display(y)),
        }
    }

    fn var(&mut self, t: impl Into<String>) -> FormulaIdx {
        let t = t.into();
        let len = VarIdx(self.vars.len());
        let var_idx = match self.vars.entry(t.clone()) {
            Occupied(o) => *o.get(),
            Vacant(v) => {
                self.var_names.push(t);
                v.insert(len);
                len
            }
        };
        let formula = Formula::Var(var_idx);
        let len = FormulaIdx(self.formulas.len());
        self.formulas.push(formula);
        len
    }

    fn and(&mut self, t1: FormulaIdx, t2: FormulaIdx) -> FormulaIdx {
        let formula = Formula::And(t1, t2);
        let len = FormulaIdx(self.formulas.len());
        self.formulas.push(formula);
        len
    }

    fn or(&mut self, t1: FormulaIdx, t2: FormulaIdx) -> FormulaIdx {
        let formula = Formula::Or(t1, t2);
        let len = FormulaIdx(self.formulas.len());
        self.formulas.push(formula);
        len
    }
    fn implies(&mut self, t1: FormulaIdx, t2: FormulaIdx) -> FormulaIdx {
        let formula = Formula::Implies(t1, t2);
        let len = FormulaIdx(self.formulas.len());
        self.formulas.push(formula);
        len
    }
    fn not(&mut self, t1: FormulaIdx) -> FormulaIdx {
        let formula = Formula::Not(t1);
        let len = FormulaIdx(self.formulas.len());
        self.formulas.push(formula);
        len
    }
    fn eq(&self, t1: FormulaIdx, t2: FormulaIdx) -> bool {
        if t1 == t2{ return true;}
        match (self[t1], self[t2]) {
            (Formula::Var(v1), Formula::Var(v2)) => v1 == v2,
            (Formula::Not(t1), Formula::Not(t2)) => self.eq(t1, t2),
            (Formula::And(l1, r1), Formula::And(l2, r2))
            | (Formula::Or(l1, r1), Formula::Or(l2, r2))
            | (Formula::Implies(l1, r1), Formula::Implies(l2, r2)) => {
                self.eq(l1, l2) && self.eq(r1, r2)
            }
            (_, _) => false,
        }
    }

    fn clear(&mut self)  {
        self.formulas.clear();
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
struct TableuxIdx(usize);

#[derive(Clone, Debug)]
struct TableuxElement {
    parent: Option<TableuxIdx>,
    sign: bool,
    formula: FormulaIdx,
}

impl Index<TableuxIdx> for Tableux {
    type Output = TableuxElement;

    fn index(&self, index: TableuxIdx) -> &Self::Output {
        &self.elements[index.0]
    }
}

impl IndexMut<TableuxIdx> for Tableux {
    fn index_mut(&mut self, index: TableuxIdx) -> &mut Self::Output {
        &mut self.elements[index.0]
    }
}

impl Index<VarIdx> for Tableux {
    type Output = String;

    fn index(&self, index: VarIdx) -> &Self::Output {
        &self.formulas[index]
    }
}
impl Index<FormulaIdx> for Tableux {
    type Output = Formula;

    fn index(&self, index: FormulaIdx) -> &Self::Output {
        &self.formulas[index]
    }
}

#[derive(Clone, Debug,Default)]
struct Tableux {
    formulas: FormulaPool,
    elements: Vec<TableuxElement>,
    children: HashMap<TableuxIdx, (Option<TableuxIdx>, Option<TableuxIdx>)>,
    hasnt_processed: HashSet<TableuxIdx>,
    open: Vec<TableuxIdx>,
    closed_branches: Vec<(TableuxIdx, TableuxIdx)>,
    cant_progress: HashSet<TableuxIdx>,
}

impl Tableux {
    fn hypothesis(formulas: FormulaPool, formula: FormulaIdx) -> Self {
        let elem = TableuxElement {
            parent: None,
            sign: false,
            formula,
        };
        let elements = vec![elem];
        let open = vec![TableuxIdx(0)];
        let children = HashMap::new();
        let closed_branches = vec![];
        let cant_progress = HashSet::new();
        let hasnt_processed = HashSet::new();
        Tableux {
            formulas,
            elements,
            children,
            hasnt_processed,
            open,
            closed_branches,
            cant_progress,
        }
    }

    fn get_parents_idx(&self, idx: TableuxIdx) -> Vec<TableuxIdx> {
        let mut ret = vec![];
        let mut idx = idx;
        loop {
            let elem = &self[idx];
            let elem_parent = elem.parent;
            ret.push(idx);
            match elem_parent {
                Some(x) => {
                    idx = x;
                }
                None => {
                    break;
                }
            }
        }
        ret
    }
    fn get_parents(&self, idx: TableuxIdx) -> Vec<TableuxElement> {
        let mut ret = vec![];
        let mut idx = idx;
        loop {
            let elem = &self[idx];
            let elem_parent = elem.parent;
            ret.push(elem.clone());
            match elem_parent {
                Some(x) => {
                    idx = x;
                }
                None => {
                    break;
                }
            }
        }
        ret
    }

    fn closes(&self, elem: &TableuxElement, path: &Vec<TableuxElement>) -> bool {
        path.iter()
            .any(|x| x.sign != elem.sign && self.formulas.eq(x.formula, elem.formula))
    }

    fn closes_opt(
        &self,
        elem: &TableuxElement,
        path: &Vec<TableuxElement>,
        path_idx: &Vec<TableuxIdx>,
    ) -> Option<TableuxIdx> {
        path.iter()
            .zip(path_idx)
            .find(|(x, _idx)| x.sign != elem.sign && self.formulas.eq(x.formula, elem.formula))
            .map(|(_, idx)| idx)
            .copied()
    }

    fn proove(&mut self, mut n: usize) {
        loop {
            println!("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*");
            if n == 0 || self.open.is_empty() {
                break;
            } else {
                n -= 1;
                self.step();
                println!("{}", &self.display());

                self.process_unprocessed_nodes();
            }
        }
    }
    fn step(&mut self) {
        let Some(idx) = self.open.pop() else {
            return;
        };
        let TableuxElement {
            parent: _,
            sign,
            formula,
        } = self[idx];
        let path = self.get_parents(idx);
        let path_idx = self.get_parents_idx(idx);
        match (sign, self[formula]) {
            (false, Formula::Implies(x, y)) => self.alpha(x, y, idx, true, false, &path, &path_idx),
            (true, Formula::And(x, y)) | (false, Formula::Or(x, y)) => {
                self.alpha(x, y, idx, sign, sign, &path, &path_idx);
            }
            (true, Formula::Or(x, y)) | (false, Formula::And(x, y)) => {
                self.beta(x, y, idx, sign, sign, &path, &path_idx);
            }
            (true, Formula::Implies(x, y)) => {
                self.beta(x, y, idx, false, true, &path, &path_idx);
            }
            (s, Formula::Not(x)) => {
                let elem = TableuxElement {
                    parent: Some(idx),
                    sign: !s,
                    formula: x,
                };
                let len = TableuxIdx(self.elements.len());
                self.children.insert(idx, (Some(len), None));
                self.elements.push(elem.clone());

                self.closes_opt(&elem, &path, &path_idx)
                    .map(|x| {
                        self.closed_branches.push((len, x));
                    })
                    .or_else(|| {
                        self.open.push(len);
                        None
                    });
            }
            (_, Formula::Var(_)) => {
                self.closes_opt(&self[idx], &path, &path_idx)
                    .map(|x| {
                        self.closed_branches.push((idx, x));
                    })
                    .or_else(|| {
                        self.cant_progress.insert(idx);
                        None
                    });
            }
        }
    }

    fn alpha(
        &mut self,
        x: FormulaIdx,
        y: FormulaIdx,
        idx: TableuxIdx,
        sign1: bool,
        sign2: bool,
        path: &Vec<TableuxElement>,
        path_idx: &Vec<TableuxIdx>,
    ) {
        let elem1 = TableuxElement {
            parent: Some(idx),
            sign: sign1,
            formula: x,
        };
        let len1 = TableuxIdx(self.elements.len());

        self.elements.push(elem1.clone());
        let elem2 = TableuxElement {
            parent: Some(len1),
            sign: sign2,
            formula: y,
        };
        let len2 = TableuxIdx(self.elements.len());
        self.elements.push(elem2.clone());

        self.children.insert(idx, (Some(len1), None));
        self.hasnt_processed.insert(len1);
        self.children.insert(len1, (Some(len2), None));

        self.closes_opt(&elem2, path, path_idx)
            .map(|x| {
                self.closed_branches.push((len2, x));
            })
            .or_else(|| {
                self.open.push(len2);
                None
            });
    }

    fn beta(
        &mut self,
        x: FormulaIdx,
        y: FormulaIdx,
        idx: TableuxIdx,
        sign1: bool,
        sign2: bool,
        path: &Vec<TableuxElement>,
        path_idx: &Vec<TableuxIdx>,
    ) {
        let elem = TableuxElement {
            parent: Some(idx),
            sign: sign1,
            formula: x,
        };
        let len1 = TableuxIdx(self.elements.len());

        self.elements.push(elem.clone());
        self.closes_opt(&elem, path, path_idx)
            .map(|x| {
                self.closed_branches.push((len1, x));
            })
            .or_else(|| {
                self.open.push(len1);
                None
            });

        let elem = TableuxElement {
            parent: Some(idx),
            sign: sign2,
            formula: y,
        };
        let len2 = TableuxIdx(self.elements.len());
        self.elements.push(elem.clone());
        self.closes_opt(&elem, path, path_idx)
            .map(|x| {
                self.closed_branches.push((len2, x));
            })
            .or_else(|| {
                self.open.push(len2);
                None
            });

        self.children.insert(idx, (Some(len1), Some(len2)));
    }

    fn display(&self) -> String {
        let mut stack = vec![(TableuxIdx(0), 2)];
        let mut ret = String::new();

        if self.open.is_empty(){
            if self.cant_progress.is_empty() {
                ret.push_str("TAUTOLOGY\n");
            }else{
                ret.push_str("NOT COMPLETE\n")
            }
        }
        ret.push_str("Tree:\n");

        loop {
            let Some((idx, depth)) = stack.pop() else {
                break;
            };
            match self.children.get(&idx) {
                Some((Some(x), Some(y))) => {
                    stack.push((*x, depth + 2));
                    stack.push((*y, depth + 2));
                }
                Some((Some(x), None)) | Some((None, Some(x))) => {
                    stack.push((*x, depth));
                }
                Some((None, None)) | None => {}
            }
            let TableuxElement {
                parent: _,
                sign,
                formula,
            } = self[idx];

            let cant_progress = if self.cant_progress.contains(&idx) { " S "} else{ "   "};
            let open = if self.open.contains(&idx) {" O "} else {"   "};
            let closed = self.closed_branches.iter().find(|x| x.0 == idx)
                .map(|x| format!(" X @ {: >3} ",x.1.0))
                .unwrap_or("         ".to_string());
                //if self.closed_branches.contains(&idx) {"X"} else {" "};
            let not_process = if self.hasnt_processed.contains(&idx) {" P "} else { "   "};
            ret.push_str(format!("\t|{: >3}\t\t|{open}|{closed}|{cant_progress}|{not_process}|\t\t",idx.0).as_str());
            for _ in 0..depth {
                ret.push('\t');
            }
            if sign {
                ret.push('T')
            } else {
                ret.push('F')
            };
            ret.push_str(" [ ");
            let f = self.formulas.display(formula);
            ret.push_str(f.as_str());
            ret.push_str(" ] \n");
            
        }
        ret.push_str("\n\n");
        //ret.push_str(format!("closed branches : {}\n", self.closed_branches.len()).as_str());
        //ret.push_str(format!("cant progress   : {}\n", self.cant_progress.len()).as_str());
        //ret.push_str(format!("open            : {}\n", self.open.len()).as_str());

        ret
    }

    fn process_unprocessed_nodes(&mut self) {
        /*let Some(idx) = self.hasnt_processed.iter().next() else { return None };*/
        if self.open.is_empty() {
            let mut indices = HashSet::new();
            for idx in self.hasnt_processed.iter() {
                let leafs: Vec<TableuxIdx> = self.get_leafs(*idx);
                let mut hasnt_processed = true;
                for leaf in leafs {
                    if self.cant_progress.contains(&leaf) {
                        hasnt_processed = false;
                        self.cant_progress.remove(&leaf);
                        let TableuxElement {
                            parent: _,
                            sign,
                            formula,
                        } = self[*idx];
                        let elem = TableuxElement {
                            parent: Some(leaf),
                            sign,
                            formula,
                        };

                        let len = TableuxIdx(self.elements.len());
                        self.children.insert(leaf, (Some(len), None));
                        self.elements.push(elem);
                        self.open.push(len);
                    }

                    /*if self.open.contains(&leaf)  {
                        hasnt_processed = false;
                        self.open = self.open.clone().into_iter().filter_map(|x| if x == leaf { None} else { Some(x)}).collect() ;
                        indices.insert(leaf);
                        let TableuxElement {
                            parent: _,
                            sign,
                            formula,
                        } = self[*idx];
                        let elem = TableuxElement {
                            parent: Some(leaf),
                            sign,
                            formula,
                        };

                        let len = TableuxIdx(self.elements.len());
                        self.children.insert(leaf, (Some(len), None));
                        self.elements.push(elem);
                        self.open.push(len);
                    }*/
                }

                if hasnt_processed {
                    indices.insert(*idx);
                }
            }
            self.hasnt_processed = indices;
        }
        /*if self.open.is_empty(){
        let mut hasnt_progressed = HashSet::new();
        let mut processed = vec![];
        for stuck in self.cant_progress.iter(){
            let mut succ = true;
            for ancestor in self.get_parents_idx(*stuck){
                if self.hasnt_processed.contains(&ancestor){
                    processed.push(ancestor);
                    let TableuxElement { parent:_, sign, formula } = self[ancestor];
                    let elem = TableuxElement{parent:Some(*stuck),sign,formula};

                    let len = TableuxIdx(self.elements.len());
                    self.children.insert(*stuck, (Some(len),None));
                    self.elements.push(elem);
                    self.open.push(len);
                    succ = false;
                    //break;


                }
            }
            if succ{ hasnt_progressed.insert(*stuck);}
        }
        self.cant_progress = hasnt_progressed;
        for elem in processed{
            self.hasnt_processed.remove(&elem);
        }
        }
        None*/
    }

    fn get_leafs(&self, idx: TableuxIdx) -> Vec<TableuxIdx> {
        let mut ret = vec![];
        let mut stack = vec![idx];
        loop {
            let Some(id) = stack.pop() else {
                break;
            };
            match self.children.get(&id) {
                Some((Some(x), Some(y))) => {
                    stack.push(*x);
                    stack.push(*y);
                }
                Some((Some(x), None)) | Some((None, Some(x))) => {
                    stack.push(*x);
                }
                Some((None, None)) | None => ret.push(id),
            }
        }
        ret
    }

    fn clear(&mut self) {
        self.formulas.clear();
        self.elements.clear();
        self.children.clear();
        self.hasnt_processed.clear();
        self.open.clear();
        self.closed_branches.clear();
        self.cant_progress.clear();
        
    }

    fn parse_hypothesis(&mut self, formula: FormulaIdx)  {
        let elem = TableuxElement {
            parent: None,
            sign: false,
            formula,
        };
        let len = TableuxIdx(self.elements.len());
        self.elements.push(elem);
        self.open.push(len);
    }
}

use pest::Parser;
use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "formula.pest"]
pub struct FormulaParser;
fn repl_helper(table:&mut Tableux,p:&str){
    table.clear();
    let ret = FormulaParser::parse(Rule::formula, p).unwrap();
    let p = parse_lisp(
        &mut table.formulas,
        ret.into_iter()
            .next()
            .unwrap()
            .into_inner()
            .next()
            .unwrap(),
    );
    table.parse_hypothesis(p);
    table.proove(1200);
}

fn repl()  {
    let mut table = Tableux::default();
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        repl_helper(&mut table, line.unwrap().as_str());
        let s = table.display();
        println!("{}", s);
        println!("Waiting for Input ...\n>> ");
    }
}


#[test]
fn parse_pairs() {
    let mut pool = FormulaPool::default();
    let ret = FormulaParser::parse(Rule::formula, "(implies ( not ( and P Q ) ) ( or  (not P Q ) ) )").unwrap();
    let p = parse_lisp(
        &mut pool,
        ret.into_iter()
            .next()
            .unwrap()
            .into_inner()
            .next()
            .unwrap(),
    );
    let mut table = Tableux::hypothesis(pool, p);
    table.proove(1000);
    println!("{}",table.display());
    //dbg!(ret);
}

fn parse_lisp(pool: &mut FormulaPool, p: pest::iterators::Pair<'_, Rule>) -> FormulaIdx {
    //dbg!(&p);
    match p.as_rule() {
        Rule::mon_call => parse_lisp_mono(pool, p.into_inner())[0],
        //Rule::name => parse_lisp_name(pool, p.into_inner()),
        Rule::bin_call => parse_lisp_bin(pool, p.into_inner())[0],

        x => unreachable!("this shouldnt happen . {:?}",x),
    }
}

fn parse_lisp_mono(
    pool: &mut FormulaPool,
    mut into_inner: pest::iterators::Pairs<'_, Rule>,
) -> Vec<FormulaIdx> {
    let _ = into_inner.next().unwrap();
    let args: Vec<_> = into_inner
        .map(|x| parse_lisp_atom(pool, x))
        .flat_map(|x| x.into_iter())
        .fuse()
        .collect::<Box<[_]>>()
        .iter()
        .map(|x| pool.not(*x))
        .collect();
    args
}

fn parse_lisp_bin(
    pool: &mut FormulaPool,
    mut into_inner: pest::iterators::Pairs<'_, Rule>,
) -> Vec<FormulaIdx> {
    let op = dbg!(into_inner.next().unwrap());
    let mut arg0 = parse_lisp_atom(pool, into_inner.next().unwrap()).into_iter();
    let init = arg0.next().unwrap();
    let args = arg0
        .chain(
            (into_inner
                .map(|x| parse_lisp_atom(pool, dbg!(x))))
            .flat_map(|x| x.into_iter()),
        )
        .fuse()
        .collect::<Box<[_]>>()
        .iter()
        .fold(init, |init,arg| {
            match op.as_rule() {
                Rule::larrow => pool.implies(init, *arg),
                Rule::land => pool.and(init, *arg),
                Rule::lor => pool.or(init, *arg),
                Rule::liff => {
                    let l = pool.implies(init, *arg);
                    let r = pool.implies(init, *arg);
                    pool.and(l, r)
                }
                x => unreachable!("{:?}", x),
            }
        });

    vec![args]
}

fn parse_lisp_atom(pool: &mut FormulaPool, x: pest::iterators::Pair<'_, Rule>) -> Vec<FormulaIdx> {
    match x.as_rule() {
        Rule::bin_call => parse_lisp_bin(pool, x.into_inner()),
        Rule::mon_call => parse_lisp_mono(pool, x.into_inner()),
        Rule::name => vec![parse_lisp_name_str(pool, x.as_str())],
        Rule::lisp_call => vec![parse_lisp(pool, x)],
        Rule::lisp_atom => parse_lisp_atom(pool, x.into_inner().next().unwrap()),
        x => unreachable!("{:?}", x),
    }
}

fn parse_lisp_name_str(
    pool: &mut FormulaPool,
    into_inner: &str,
) -> FormulaIdx {
    dbg!(&into_inner);
    pool.var(into_inner)
}

fn main() {
    /*let mut pool = FormulaPool::default();
    let p = pool.var("p");
    let q = pool.var("q");
    let r = pool.var("r");
    let qandr = pool.and(q, r);

    let p_or_qandr = pool.or(p, qandr);

    let porq = pool.or(p, q);
    let porr = pool.or(p, r);
    let temp = pool.and(porq, porr);
    let fin = pool.implies(p_or_qandr, temp);
    let not_fin = pool.not(fin);
    let fin_and_not_fin = pool.and(fin, not_fin);
    let fin_or_not_fin = pool.or(fin, not_fin);
    let mut tableux = Tableux::hypothesis(pool, fin);

    tableux.proove(3000);
    let s = tableux.display();
    //dbg!(&tableux);
    println!("{}", s);

     */
    repl()
}
