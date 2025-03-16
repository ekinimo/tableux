mod common;

use std::io::{self, BufRead};
use common::{Tableux, FormulaPool, FormulaParser, Rule, parse_lisp};
use pest::Parser;

fn repl_helper(table: &mut Tableux, p: &str) {
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

fn repl() {
    let mut table = Tableux::default();
    let stdin = io::stdin();
    println!("Tableaux REPL - Enter a formula:");
    println!(">> ");
    for line in stdin.lock().lines() {
        repl_helper(&mut table, line.unwrap().as_str());
        let s = table.display();
        println!("{}", s);
        println!("Waiting for Input ...\n>> ");
    }
}

fn main() {
    repl()
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
    println!("{}", table.display());
}
