# [Method of Analytical Tableux](https://en.wikipedia.org/wiki/Method_of_analytic_tableaux#Clause_tableaux)
a toy implementation of the logical resolution method for learning purposes.
# Roadplan
- [issue]   infix and forthfix parser-builders
- [issue]   error handling
- [feature] Heuristics for which open branch to use
  - Interactively Select which rule to expand
  - rule size
  - \alpha \beta \gamma \delta rule order
  - iterative deepening
- [feature] First Order Logic with Unification
- [feature] Better visualization

# Examples
```
( iff ( not ( and P Q R ) ) ( or ( not P Q R ) ) )
( iff ( not ( and P Q R S T ) ) ( or ( not P Q R S T ) ) )
```
# Gotchas
Current lisp parser will map unary operator (`not`) if it has more than one argument. If there is no surrounding call first element of the list will be negated.
Binary operators will be folded if it has more than 2 arguments. 
I.e. `(not A B C D)` will result in `(not A)` and `(and (not A B C D))` will result in `(and (not A) (not B) (not C) (not D))`.
