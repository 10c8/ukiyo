# ukiyo

An interpreted language for superpowered LLM inference.

## TODOs

- [ ] Implement lists
- [ ] Add arguments to expression definitions

## Grammar

```ebnf
program = { newline }, stmt, { newline, stmt }

stmt = expr_def | expr_call | string | number

expr_def = ident, { ident }, '->', (block | case | each | expr)
expr_call = term, { expr_call }

expr = [ '(' ], ident, { expr_tail }, [ ')' ]
expr_tail = { expr_apl | scope_binding | expr_call_args }
expr_apl = '<|', expr
scope_binding = '>>', expr
expr_call_args = { term }

block = indent, block_stmt, { newline, block_stmt }, dedent
block_stmt = expr_def | scope_binding | case | each | expr | string | number

case = 'case', expr, 'of', case_block
case_block = indent, case_branch, { newline, case_branch }, dedent
case_branch = term | '_', '=>', expr

each = 'each', term | range, 'do', block | expr_call | string | number

range = integer, '..=' | '..<', integer

term = ident | string | number | list

ident = lowercase | '_', { lowercase | '_' }
string = ('"', { any }, '"') | ('"""', newline, { any }, newline, '"""')
number = integer | float
integer = digit, { digit }
float = integer, '.', integer
list = '[', [ term, { ',', term } ], ']'
```
