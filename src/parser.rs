use crate::lexer::*;

#[derive(Debug)]
pub enum ParseError {
    UnexpectedToken(&'static str, Token, usize),
    UnexpectedIndent(usize, usize),
    NumberParseError(String, usize),
    EmptyBlock(usize),
    EmptyCase(usize),
}

#[derive(serde::Serialize, Debug, Clone, PartialEq)]
pub enum Node {
    Program {
        children: Vec<Node>,
    },
    Block {
        children: Vec<Node>,
    },
    ExpressionDefinition {
        name: String,
        expression: Box<Node>,
    },
    ExpressionCall {
        name: String,
        arguments: Vec<Node>,
    },
    ExpressionApplication {
        left: Box<Node>,
        right: Box<Node>,
    },
    ScopeBinding {
        left: Option<Box<Node>>,
        right: Box<Node>,
    },
    Case {
        term: Box<Node>,
        children: Vec<Node>,
    },
    CaseBranch {
        pattern: Option<Box<Node>>,
        result: Box<Node>,
    },
    Each {
        collection: Box<Node>,
        children: Vec<Node>,
    },
    Identifier(&'static str),
    String(&'static str),
    Integer(u64),
    Float(f64),
    List {
        children: Vec<Node>,
    },
    Range {
        is_exclusive: bool,
        start: Box<Node>,
        end: Box<Node>,
    },
}

const SCOPE_BIND_PRECEDENCE: usize = 1;
const EXPR_APL_PRECEDENCE: usize = 2;

pub struct Parser {
    lexer: Lexer,
    indent_stack: Vec<usize>,
    tab_size: usize,
}

impl Parser {
    pub fn new(lexer: Lexer) -> Self {
        let indent_stack = Vec::new();

        Parser {
            lexer,
            indent_stack,
            tab_size: 0,
        }
    }

    pub fn parse(&mut self) -> Result<Vec<Node>, ParseError> {
        let program = self.program()?;

        if self.lexer.peek() != Token::EOF {
            return Err(ParseError::UnexpectedToken(
                "program",
                self.lexer.peek(),
                self.lexer.cursor(),
            ));
        }

        match program {
            Node::Program { children } => Ok(children),
            _ => unreachable!(),
        }
    }

    /// If the next token if it matches `token`, consumes it if `consume` is true.
    /// Otherwise, returns an error.
    fn require(&mut self, token: Token, consume: bool) -> Result<(), ParseError> {
        if self.lexer.peek() == token {
            if consume {
                self.lexer.next();
            }

            Ok(())
        } else {
            Err(ParseError::UnexpectedToken(
                Box::leak(format!("require({:?})", token).into_boxed_str()),
                self.lexer.peek(),
                self.lexer.cursor(),
            ))
        }
    }

    /// Consumes the next token if it is a valid indent and updates the indent stack.
    /// Otherwise, returns an error.
    fn require_indent(&mut self) -> Result<(), ParseError> {
        let last_indent = self.indent_stack.last().unwrap_or(&0).clone();

        match self.lexer.peek() {
            Token::Indent(spaces) => {
                if self.tab_size == 0 {
                    self.tab_size = spaces;
                }

                if spaces == last_indent + self.tab_size {
                    self.lexer.next();
                    self.indent_stack.push(spaces);
                } else {
                    return Err(ParseError::UnexpectedIndent(spaces, self.lexer.cursor()));
                }
            }
            _ => {
                return Err(ParseError::UnexpectedToken(
                    "indent",
                    self.lexer.peek(),
                    self.lexer.cursor(),
                ));
            }
        }

        Ok(())
    }

    /// Consumes the next token if it is a valid dedent and updates the indent stack.
    /// Otherwise, returns an error.
    fn require_dedent(&mut self) -> Result<(), ParseError> {
        let last_indent = self.indent_stack.pop().unwrap_or(0);
        self.require(Token::Dedent(last_indent), true)?;

        Ok(())
    }

    /// Consumes all newlines (and optionally indents and dedents) until a non-newline token is
    /// found.
    fn consume_newlines(&mut self, consume_indents: bool) {
        while match self.lexer.peek() {
            Token::Newline => true,
            Token::Indent(_) | Token::Dedent(_) => consume_indents,
            _ => false,
        } {
            self.lexer.next();
        }
    }

    // Node parsers
    fn program(&mut self) -> Result<Node, ParseError> {
        // program = SOI '\n'* stmt ('\n' stmt)* EOI

        let mut children = Vec::new();

        loop {
            match self.lexer.peek() {
                Token::Newline => {
                    self.lexer.next();
                }
                Token::Identifier(_) | Token::String(_) | Token::Number(_) => {
                    children.push(self.statement()?);

                    match self.lexer.peek() {
                        Token::Identifier(_) | Token::String(_) | Token::Number(_) => {
                            self.require(Token::Newline, true)?;
                        }
                        _ => {}
                    }
                }
                Token::EOF => break,
                _ => {
                    return Err(ParseError::UnexpectedToken(
                        "program",
                        self.lexer.peek(),
                        self.lexer.cursor(),
                    ))
                }
            }
        }

        Ok(Node::Program { children })
    }

    fn statement(&mut self) -> Result<Node, ParseError> {
        // stmt = expr_def | expr_call | string | number

        let result = match self.lexer.peek() {
            Token::Identifier(_) => match self.expression_definition() {
                Ok(node) => node,
                Err(_) => {
                    self.lexer.backtrack();

                    let name = match self.term() {
                        Ok(Node::Identifier(name)) => name.to_string(),
                        _ => {
                            return Err(ParseError::UnexpectedToken(
                                "expr_call",
                                self.lexer.peek(),
                                self.lexer.cursor(),
                            ));
                        }
                    };
                    let arguments = self.expression_call_arguments().unwrap_or(Vec::new());

                    Node::ExpressionCall { name, arguments }
                }
            },
            Token::String(_) | Token::Number(_) => self.term()?,
            _ => {
                return Err(ParseError::UnexpectedToken(
                    "statement",
                    self.lexer.peek(),
                    self.lexer.cursor(),
                ))
            }
        };

        Ok(result)
    }

    fn expression_definition(&mut self) -> Result<Node, ParseError> {
        // expr_def = identifier '->' (expr | case | each | block)

        let name = match self.lexer.next() {
            Token::Identifier(name) => name,
            _ => unreachable!("expr_def must start with an identifier"),
        };

        self.require(Token::AssignmentArrow, true)?;

        let expression = match self.lexer.peek() {
            Token::Identifier(_) | Token::Symbol('(') => self.expression(None)?,
            Token::String(_) | Token::Number(_) | Token::Symbol('[') => self.term()?,
            Token::Keyword(Keyword::Case) => self.case()?,
            Token::Keyword(Keyword::Each) => self.each()?,
            Token::Indent(_) => self.block()?,
            _ => {
                return Err(ParseError::UnexpectedToken(
                    "expr_def",
                    self.lexer.peek(),
                    self.lexer.cursor(),
                ))
            }
        };

        Ok(Node::ExpressionDefinition {
            name: name.to_string(),
            expression: Box::new(expression),
        })
    }

    fn expression_call_arguments(&mut self) -> Result<Vec<Node>, ParseError> {
        // expr_call = ident term*

        match self.lexer.peek() {
            Token::Identifier(_) | Token::String(_) | Token::Number(_) => {
                let mut arguments = Vec::new();

                loop {
                    match self.lexer.peek() {
                        Token::Identifier(_) | Token::String(_) | Token::Number(_) => {
                            arguments.push(self.term()?)
                        }
                        _ => break,
                    }
                }

                return Ok(arguments);
            }
            _ => {
                return Err(ParseError::UnexpectedToken(
                    "expr_call",
                    self.lexer.peek(),
                    self.lexer.cursor(),
                ))
            }
        };
    }

    fn expression(&mut self, precedence: Option<usize>) -> Result<Node, ParseError> {
        // expr = '('? ident expr_tail ')'?
        // expr_tail = (expr_apl | scope_binding | expr_call_args)*
        // expr_apl = '<|' expr
        // scope_binding = '>>' expr
        // expr_call_args = term*

        let precedence = precedence.unwrap_or(0);
        let mut is_parens = false;

        if self.lexer.peek() == Token::Symbol('(') {
            is_parens = true;
            self.lexer.next();
        }

        let left = self.identifier()?;

        let mut current = match left {
            Node::Identifier(name) => Node::ExpressionCall {
                name: name.to_string(),
                arguments: Vec::new(),
            },
            _ => unreachable!("expr must start with an ident"),
        };

        loop {
            match self.lexer.peek() {
                Token::ApplicationArrow => {
                    if !is_parens && precedence >= EXPR_APL_PRECEDENCE {
                        break;
                    }

                    self.lexer.next();

                    let right = self.expression(Some(EXPR_APL_PRECEDENCE))?;

                    current = Node::ExpressionApplication {
                        left: Box::new(current),
                        right: Box::new(right),
                    };
                }
                Token::BindingArrow => {
                    if !is_parens && precedence >= SCOPE_BIND_PRECEDENCE {
                        break;
                    }

                    self.lexer.next();

                    let right = self.expression(Some(SCOPE_BIND_PRECEDENCE))?;

                    current = Node::ScopeBinding {
                        left: Some(Box::new(current)),
                        right: Box::new(right),
                    };
                }
                Token::Identifier(_) | Token::String(_) | Token::Number(_) => {
                    let name = match current {
                        Node::Identifier(name) => name.to_string(),
                        Node::ExpressionCall { name, .. } => name.to_string(),
                        _ => unreachable!("expr_call must start with an identifier"),
                    };
                    let arguments = self.expression_call_arguments().unwrap_or(Vec::new());

                    current = Node::ExpressionCall { name, arguments };
                }
                _ => {
                    if is_parens {
                        self.require(Token::Symbol(')'), true)?;
                        is_parens = false;
                        continue;
                    }

                    break;
                }
            }
        }

        Ok(current)
    }

    fn identifier(&mut self) -> Result<Node, ParseError> {
        match self.lexer.next() {
            Token::Identifier(name) => Ok(Node::Identifier(name)),
            _ => {
                self.lexer.backtrack();
                return Err(ParseError::UnexpectedToken(
                    "identifier",
                    self.lexer.peek(),
                    self.lexer.cursor(),
                ));
            }
        }
    }

    fn term(&mut self) -> Result<Node, ParseError> {
        let result = match self.lexer.next() {
            Token::Identifier(name) => Node::Identifier(name),
            Token::String(string) => Node::String(string),
            Token::Number(num) => match self.lexer.peek() {
                Token::Symbol('.') => {
                    self.lexer.next();

                    let next = self.lexer.next();

                    match next {
                        Token::Number(frac) => {
                            let number = format!("{}.{}", num, frac);
                            Node::Float(number.parse().map_err(|_| {
                                ParseError::NumberParseError(number, self.lexer.cursor())
                            })?)
                        }
                        _ => {
                            return Err(ParseError::UnexpectedToken(
                                "number",
                                next,
                                self.lexer.cursor(),
                            ))
                        }
                    }
                }
                Token::RangeOperator(op) => {
                    return Err(ParseError::UnexpectedToken(
                        "number",
                        Token::RangeOperator(op),
                        self.lexer.cursor(),
                    ))
                }
                _ => Node::Integer(num),
            },
            Token::Symbol('[') => {
                self.consume_newlines(true);

                if let Token::Symbol(']') = self.lexer.peek() {
                    self.lexer.next();

                    return Ok(Node::List {
                        children: Vec::new(),
                    });
                };

                let mut children = Vec::new();

                loop {
                    self.consume_newlines(true);

                    children.push(self.term()?);

                    self.consume_newlines(true);

                    match self.lexer.peek() {
                        Token::Symbol(',') => {
                            self.lexer.next();

                            self.consume_newlines(true);

                            match self.lexer.peek() {
                                Token::Symbol(']') => {
                                    self.lexer.next();
                                    break;
                                }
                                _ => continue,
                            }
                        }
                        _ => {
                            self.require(Token::Symbol(']'), true)?;
                            break;
                        }
                    }
                }

                Node::List { children }
            }
            _ => {
                self.lexer.backtrack();
                return Err(ParseError::UnexpectedToken(
                    "term",
                    self.lexer.peek(),
                    self.lexer.cursor(),
                ));
            }
        };

        Ok(result)
    }

    fn case(&mut self) -> Result<Node, ParseError> {
        // case = 'case' term 'of' case_body
        // case_body = INDENT case_branch (NEWLINE case_branch)* DEDENT
        // case_branch = case_pattern '=>' expr | term
        // case_pattern = '_' | string | number

        self.require(Token::Keyword(Keyword::Case), true)?;

        let term = self.term()?;

        self.require(Token::Keyword(Keyword::Of), true)?;

        self.require_indent()?;

        let mut children = Vec::new();

        loop {
            let pattern = match self.term()? {
                Node::Identifier("_") => None,
                any => Some(Box::new(any)),
            };

            self.require(Token::MatchArrow, true)?;

            let result = match self.expression(None) {
                Ok(expr) => expr,
                Err(_) => self.term()?,
            };

            children.push(Node::CaseBranch {
                pattern,
                result: Box::new(result),
            });

            if self.lexer.peek() == Token::Newline {
                self.lexer.next();
            } else {
                self.require_dedent()?;
                break;
            }
        }

        if children.is_empty() {
            return Err(ParseError::EmptyCase(self.lexer.cursor()));
        }

        Ok(Node::Case {
            term: Box::new(term),
            children,
        })
    }

    fn each(&mut self) -> Result<Node, ParseError> {
        // each = 'each' (term | range) 'do' (expr_call | string | number | block)

        self.require(Token::Keyword(Keyword::Each), true)?;

        let collection = match self.lexer.peek() {
            Token::Identifier(_) | Token::String(_) => self.term()?,
            Token::Number(_) => match self.range() {
                Ok(range) => range,
                Err(_) => {
                    self.lexer.backtrack();
                    self.term()?
                }
            },
            _ => {
                return Err(ParseError::UnexpectedToken(
                    "each",
                    self.lexer.peek(),
                    self.lexer.cursor(),
                ))
            }
        };

        self.require(Token::Keyword(Keyword::Do), true)?;

        let children = match self.lexer.peek() {
            Token::Indent(_) => match self.block()? {
                Node::Block { children } => children,
                _ => unreachable!("block must be a block (duh?)"),
            },
            _ => {
                let child = self.expression(None)?;
                vec![child]
            }
        };

        Ok(Node::Each {
            collection: Box::new(collection),
            children,
        })
    }

    fn block(&mut self) -> Result<Node, ParseError> {
        // block = INDENT block_stmt (block_stmt NEWLINE)+ DEDENT
        // block_stmt = expr_def | block_scope_binding | expr | case | each

        self.require_indent()?;

        let mut children = Vec::new();

        loop {
            let child = match self.lexer.peek() {
                Token::Identifier(_) => match self.expression_definition() {
                    Ok(node) => node,
                    Err(_) => self.expression(None)?,
                },
                Token::String(_) | Token::Number(_) => self.term()?,
                Token::BindingArrow => {
                    self.lexer.next();

                    let right = self.expression(None)?;

                    Node::ScopeBinding {
                        left: None,
                        right: Box::new(right),
                    }
                }
                Token::Keyword(Keyword::Case) => self.case()?,
                Token::Keyword(Keyword::Each) => self.each()?,
                _ => {
                    return Err(ParseError::UnexpectedToken(
                        "block_stmt",
                        self.lexer.peek(),
                        self.lexer.cursor(),
                    ))?;
                }
            };

            children.push(child);

            match self.lexer.peek() {
                Token::Dedent(_) => {
                    self.require_dedent()?;
                    break;
                }
                Token::Newline => {
                    self.lexer.next();
                }
                _ => {
                    return Err(ParseError::UnexpectedToken(
                        "block_stmt",
                        self.lexer.peek(),
                        self.lexer.cursor(),
                    ))
                }
            }
        }

        if children.is_empty() {
            return Err(ParseError::EmptyBlock(self.lexer.cursor()));
        }

        Ok(Node::Block { children })
    }

    fn range(&mut self) -> Result<Node, ParseError> {
        let start = match self.lexer.next() {
            Token::Number(num) => Node::Integer(num),
            _ => {
                self.lexer.backtrack();
                return Err(ParseError::UnexpectedToken(
                    "range",
                    self.lexer.peek(),
                    self.lexer.cursor(),
                ));
            }
        };

        let op = match self.lexer.next() {
            Token::RangeOperator(op) => op,
            _ => {
                self.lexer.backtrack();
                return Err(ParseError::UnexpectedToken(
                    "range",
                    self.lexer.peek(),
                    self.lexer.cursor(),
                ));
            }
        };

        let end = match self.lexer.next() {
            Token::Number(num) => Node::Integer(num),
            _ => {
                self.lexer.backtrack();
                return Err(ParseError::UnexpectedToken(
                    "range",
                    self.lexer.peek(),
                    self.lexer.cursor(),
                ));
            }
        };

        Ok(Node::Range {
            is_exclusive: op == RangeOperator::Exclusive,
            start: Box::new(start),
            end: Box::new(end),
        })
    }
}

// TESTS
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block() {
        let mut lexer = Lexer::new(
            r#"
block ->
  "one"
  "two"
"three"
"#,
        );
        lexer.lex().expect("failed to lex");

        let mut parser = Parser::new(lexer);
        let ast = parser.parse().expect("failed to parse");

        assert_eq!(
            ast,
            [
                Node::ExpressionDefinition {
                    name: "block".to_string(),
                    expression: Box::new(Node::Block {
                        children: [Node::String("one"), Node::String("two")].to_vec()
                    }),
                },
                Node::String("three"),
            ]
        );
    }

    #[test]
    fn test_expr_def() {
        let mut lexer = Lexer::new(r#"hello -> "world""#);
        lexer.lex().expect("failed to lex");

        let mut parser = Parser::new(lexer);
        let ast = parser.parse().expect("failed to parse");

        assert_eq!(
            ast,
            [Node::ExpressionDefinition {
                name: "hello".to_string(),
                expression: Box::new(Node::String("world")),
            }]
        );

        lexer = Lexer::new(
            r#"
hello ->
  "world"
  "world"
"#,
        );
        lexer.lex().expect("failed to lex");

        parser = Parser::new(lexer);
        let ast = parser.parse().expect("failed to parse");

        assert_eq!(
            ast,
            [Node::ExpressionDefinition {
                name: "hello".to_string(),
                expression: Box::new(Node::Block {
                    children: [Node::String("world"), Node::String("world")].to_vec()
                }),
            }]
        );

        lexer = Lexer::new(r#"hello -> "world" "world""#);
        lexer.lex().expect("failed to lex");

        parser = Parser::new(lexer);
        let ast = parser.parse();

        assert!(ast.is_err());
    }

    #[test]
    fn test_expr_call() {
        let mut lexer = Lexer::new(r#"hello "world" 3"#);
        lexer.lex().expect("failed to lex");

        let mut parser = Parser::new(lexer);
        let ast = parser.parse().expect("failed to parse");

        assert_eq!(
            ast,
            [Node::ExpressionCall {
                name: "hello".to_string(),
                arguments: [Node::String("world"), Node::Integer(3)].to_vec()
            }]
        );

        lexer = Lexer::new(r#"hello"#);
        lexer.lex().expect("failed to lex");

        parser = Parser::new(lexer);
        let ast = parser.parse().expect("failed to parse");

        assert_eq!(
            ast,
            [Node::ExpressionCall {
                name: "hello".to_string(),
                arguments: [].to_vec(),
            }]
        );
    }

    #[test]
    fn test_expr_apl() {
        let mut lexer = Lexer::new(r#"a -> b <| c <| d"#);
        lexer.lex().expect("failed to lex");

        let mut parser = Parser::new(lexer);
        let ast = parser.parse().expect("failed to parse");

        assert_eq!(
            ast,
            [Node::ExpressionDefinition {
                name: "a".to_string(),
                expression: Box::new(Node::ExpressionApplication {
                    left: Box::new(Node::ExpressionApplication {
                        left: Box::new(Node::ExpressionCall {
                            name: "b".to_string(),
                            arguments: [].to_vec(),
                        }),
                        right: Box::new(Node::ExpressionCall {
                            name: "c".to_string(),
                            arguments: [].to_vec(),
                        }),
                    }),
                    right: Box::new(Node::ExpressionCall {
                        name: "d".to_string(),
                        arguments: [].to_vec()
                    })
                })
            }]
        );

        let mut lexer = Lexer::new(r#"a -> b <| (c <| d)"#);
        lexer.lex().expect("failed to lex");

        let mut parser = Parser::new(lexer);
        let ast = parser.parse().expect("failed to parse");

        assert_eq!(
            ast,
            [Node::ExpressionDefinition {
                name: "a".to_string(),
                expression: Box::new(Node::ExpressionApplication {
                    left: Box::new(Node::ExpressionCall {
                        name: "b".to_string(),
                        arguments: [].to_vec(),
                    }),
                    right: Box::new(Node::ExpressionApplication {
                        left: Box::new(Node::ExpressionCall {
                            name: "c".to_string(),
                            arguments: [].to_vec()
                        }),
                        right: Box::new(Node::ExpressionCall {
                            name: "d".to_string(),
                            arguments: [].to_vec()
                        })
                    })
                })
            }]
        );
    }

    #[test]
    fn test_scope_binding() {
        let mut lexer = Lexer::new(r#"a -> b >> c"#);
        lexer.lex().expect("failed to lex");

        let mut parser = Parser::new(lexer);
        let ast = parser.parse().expect("failed to parse");

        assert_eq!(
            ast,
            [Node::ExpressionDefinition {
                name: "a".to_string(),
                expression: Box::new(Node::ScopeBinding {
                    left: Some(Box::new(Node::ExpressionCall {
                        name: "b".to_string(),
                        arguments: [].to_vec()
                    })),
                    right: Box::new(Node::ExpressionCall {
                        name: "c".to_string(),
                        arguments: [].to_vec()
                    })
                })
            }]
        );

        let mut lexer = Lexer::new(r#"a -> b <| c >> d"#);
        lexer.lex().expect("failed to lex");

        let mut parser = Parser::new(lexer);
        let ast = parser.parse().expect("failed to parse");

        assert_eq!(
            ast,
            [Node::ExpressionDefinition {
                name: "a".to_string(),
                expression: Box::new(Node::ScopeBinding {
                    left: Some(Box::new(Node::ExpressionApplication {
                        left: Box::new(Node::ExpressionCall {
                            name: "b".to_string(),
                            arguments: [].to_vec()
                        }),
                        right: Box::new(Node::ExpressionCall {
                            name: "c".to_string(),
                            arguments: [].to_vec()
                        })
                    })),
                    right: Box::new(Node::ExpressionCall {
                        name: "d".to_string(),
                        arguments: [].to_vec()
                    })
                })
            }]
        );

        let mut lexer = Lexer::new(r#"a -> b <| (c >> d)"#);
        lexer.lex().expect("failed to lex");

        let mut parser = Parser::new(lexer);
        let ast = parser.parse().expect("failed to parse");

        assert_eq!(
            ast,
            [Node::ExpressionDefinition {
                name: "a".to_string(),
                expression: Box::new(Node::ExpressionApplication {
                    left: Box::new(Node::ExpressionCall {
                        name: "b".to_string(),
                        arguments: [].to_vec()
                    }),
                    right: Box::new(Node::ScopeBinding {
                        left: Some(Box::new(Node::ExpressionCall {
                            name: "c".to_string(),
                            arguments: [].to_vec()
                        })),
                        right: Box::new(Node::ExpressionCall {
                            name: "d".to_string(),
                            arguments: [].to_vec()
                        })
                    }),
                })
            }]
        );

        let mut lexer = Lexer::new(
            r#"
a ->
  "b"
  >> c
"#,
        );
        lexer.lex().expect("failed to lex");

        let mut parser = Parser::new(lexer);
        let ast = parser.parse().expect("failed to parse");

        assert_eq!(
            ast,
            [Node::ExpressionDefinition {
                name: "a".to_string(),
                expression: Box::new(Node::Block {
                    children: [
                        Node::String("b"),
                        Node::ScopeBinding {
                            left: None,
                            right: Box::new(Node::ExpressionCall {
                                name: "c".to_string(),
                                arguments: [].to_vec()
                            })
                        }
                    ]
                    .to_vec()
                })
            }]
        );
    }

    #[test]
    fn test_case() {
        let mut lexer = Lexer::new(
            r#"
color -> case it of
  "apples"  => "red"
  "bananas" => "yellow"
  _         => "unknown"
"#,
        );
        lexer.lex().expect("failed to lex");

        let mut parser = Parser::new(lexer);
        let ast = parser.parse().expect("failed to parse");

        assert_eq!(
            ast,
            [Node::ExpressionDefinition {
                name: "color".to_string(),
                expression: Box::new(Node::Case {
                    term: Box::new(Node::Identifier("it")),
                    children: [
                        Node::CaseBranch {
                            pattern: Some(Box::new(Node::String("apples"))),
                            result: Box::new(Node::String("red"))
                        },
                        Node::CaseBranch {
                            pattern: Some(Box::new(Node::String("bananas"))),
                            result: Box::new(Node::String("yellow"))
                        },
                        Node::CaseBranch {
                            pattern: None,
                            result: Box::new(Node::String("unknown"))
                        }
                    ]
                    .to_vec()
                })
            }]
        );

        let mut lexer = Lexer::new(
            r#"
color -> case it of

"#,
        );
        lexer.lex().expect("failed to lex");

        let mut parser = Parser::new(lexer);
        let ast = parser.parse();

        assert!(ast.is_err());
    }

    #[test]
    fn test_each() {
        let mut lexer = Lexer::new(
            r#"
colors ->
  each fruits do
    color -> to_color it
"#,
        );
        lexer.lex().expect("failed to lex");

        let mut parser = Parser::new(lexer);
        let ast = parser.parse().expect("failed to parse");

        assert_eq!(
            ast,
            [Node::ExpressionDefinition {
                name: "colors".to_string(),
                expression: Box::new(Node::Block {
                    children: [Node::Each {
                        collection: Box::new(Node::Identifier("fruits")),
                        children: [Node::ExpressionDefinition {
                            name: "color".to_string(),
                            expression: Box::new(Node::ExpressionCall {
                                name: "to_color".to_string(),
                                arguments: [Node::Identifier("it")].to_vec()
                            })
                        }]
                        .to_vec()
                    }]
                    .to_vec()
                })
            }]
        );

        let mut lexer = Lexer::new(
            r#"
colors -> each fruits do

"#,
        );
        lexer.lex().expect("failed to lex");

        let mut parser = Parser::new(lexer);
        let ast = parser.parse();

        assert!(ast.is_err());
    }

    #[test]
    fn test_list() {
        let mut lexer = Lexer::new(
            r#"
colors -> [
    "red",
  3
, 4.2]
"#,
        );
        lexer.lex().expect("failed to lex");

        let mut parser = Parser::new(lexer);
        let ast = parser.parse().expect("failed to parse");

        assert_eq!(
            ast,
            [Node::ExpressionDefinition {
                name: "colors".to_string(),
                expression: Box::new(Node::List {
                    children: [Node::String("red"), Node::Integer(3), Node::Float(4.2)].to_vec()
                })
            }]
        );

        lexer = Lexer::new(r#"colors -> []"#);
        lexer.lex().expect("failed to lex");

        parser = Parser::new(lexer);
        let ast = parser.parse().expect("failed to parse");

        assert_eq!(
            ast,
            [Node::ExpressionDefinition {
                name: "colors".to_string(),
                expression: Box::new(Node::List {
                    children: [].to_vec()
                })
            }]
        );
    }
}
