use crate::lexer::*;

#[derive(Debug)]
pub enum ParseError {
    UnexpectedToken(&'static str, Token, usize),
    NumberParseError(String, usize),
    EmptyBlock(usize),
}

#[derive(Debug, Clone, PartialEq)]
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
        left: Box<Node>,
        right: Box<Node>,
    },
    Case {
        term: Box<Node>,
        children: Vec<Node>,
    },
    CaseBranch {
        pattern: Box<Node>,
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
    Range {
        is_exclusive: bool,
        start: Box<Node>,
        end: Box<Node>,
    },
}

pub struct Parser {
    lexer: Lexer,
    indent_stack: Vec<usize>,
}

impl Parser {
    pub fn new(lexer: Lexer) -> Self {
        let indent_stack = Vec::new();

        Parser {
            lexer,
            indent_stack,
        }
    }

    pub fn parse(&mut self) -> Result<Node, ParseError> {
        let program = self.program()?;

        if self.lexer.peek() != Token::EOF {
            return Err(ParseError::UnexpectedToken(
                "program",
                self.lexer.peek(),
                self.lexer.cursor(),
            ));
        }

        Ok(program)
    }

    /// Consumes the next token if it matches `token`.
    /// Otherwise, returns an error.
    fn require(&mut self, token: Token) -> Result<(), ParseError> {
        if self.lexer.peek() == token {
            self.lexer.next();
            Ok(())
        } else {
            Err(ParseError::UnexpectedToken(
                "require",
                self.lexer.peek(),
                self.lexer.cursor(),
            ))
        }
    }

    /// Consumes the next token if it is a valid indent and updates the indent stack.
    /// Otherwise, returns an error.
    fn require_indent(&mut self) -> Result<(), ParseError> {
        let last_indent = self.indent_stack.last().unwrap_or(&0).clone();
        self.require(Token::Indent(last_indent + 1))?;

        self.indent_stack.push(last_indent + 1);

        Ok(())
    }

    /// Consumes the next token if it is a valid dedent and updates the indent stack.
    /// Otherwise, returns an error.
    fn require_dedent(&mut self) -> Result<(), ParseError> {
        let last_indent = self.indent_stack.pop().unwrap_or(0);
        self.require(Token::Dedent(last_indent))?;

        Ok(())
    }

    // Node parsers
    fn program(&mut self) -> Result<Node, ParseError> {
        // program = SOI '\n'* stmt* EOI

        let mut children = Vec::new();

        loop {
            match self.lexer.peek() {
                Token::EOF => break,
                Token::Newline => {
                    self.lexer.next();
                }
                _ => children.push(self.statement()?),
            }
        }

        Ok(Node::Program { children })
    }

    fn statement(&mut self) -> Result<Node, ParseError> {
        // stmt = expr_def | term

        let result = match self.lexer.peek() {
            Token::Identifier(_) => match self.expression_definition() {
                Ok(node) => node,
                Err(_) => {
                    self.lexer.backtrack();
                    self.term()?
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
        // exprdef = identifier '->' (expr | case | each | block)

        let name = match self.lexer.next() {
            Token::Identifier(name) => name,
            _ => unreachable!("expr_def must start with an identifier"),
        };

        self.require(Token::AssignmentArrow)?;

        let expression = match self.lexer.peek() {
            Token::Identifier(_) => self.expression()?,
            Token::String(_) | Token::Number(_) => self.term()?,
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

    fn expression(&mut self) -> Result<Node, ParseError> {
        // expr = term expr_tail
        // expr_tail = (expr_apl | expr_call)*
        // expr_apl = '<|' expr
        // expr_call = ident term*

        let left = self.term()?;

        let mut current = left;

        loop {
            match self.lexer.peek() {
                Token::ApplicationArrow => {
                    self.lexer.next();

                    let right = self.expression()?;

                    current = Node::ExpressionApplication {
                        left: Box::new(current),
                        right: Box::new(right),
                    };
                }
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

                    current = Node::ExpressionCall {
                        name: match current {
                            Node::Identifier(name) => name.to_string(),
                            _ => {
                                return Err(ParseError::UnexpectedToken(
                                    "expression",
                                    self.lexer.peek(),
                                    self.lexer.cursor(),
                                ))
                            }
                        },
                        arguments,
                    };
                }
                _ => break,
            }
        }

        Ok(current)
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
            _ => {
                return Err(ParseError::UnexpectedToken(
                    "term",
                    self.lexer.peek(),
                    self.lexer.cursor(),
                ))
            }
        };

        Ok(result)
    }

    fn case(&mut self) -> Result<Node, ParseError> {
        // case = 'case' term 'of' case_body
        // case_pattern = '_' | string | number
        // case_branch = case_pattern '=>' term
        // case_body = INDENT case_branch (NEWLINE case_branch)* DEDENT

        self.require(Token::Keyword(Keyword::Case))?;

        let term = self.term()?;

        self.require(Token::Keyword(Keyword::Of))?;

        self.require_indent()?;

        let mut children = Vec::new();

        loop {
            let pattern = self.term()?;

            self.require(Token::MatchArrow)?;

            let result = self.term()?;

            children.push(Node::CaseBranch {
                pattern: Box::new(pattern),
                result: Box::new(result),
            });

            if self.lexer.peek() == Token::Newline {
                self.lexer.next();
            } else {
                self.require_dedent()?;
                break;
            }
        }

        Ok(Node::Case {
            term: Box::new(term),
            children,
        })
    }

    fn each(&mut self) -> Result<Node, ParseError> {
        // each = 'each' (term | range) 'do' block

        self.require(Token::Keyword(Keyword::Each))?;

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

        self.require(Token::Keyword(Keyword::Do))?;

        let block = if let Node::Block { children } = self.block()? {
            children
        } else {
            unreachable!("block must be a block (duh?)")
        };

        Ok(Node::Each {
            collection: Box::new(collection),
            children: block,
        })
    }

    fn block(&mut self) -> Result<Node, ParseError> {
        // block = INDENT block_stmt (block_stmt NEWLINE)+ DEDENT
        // block_stmt = expr_def | expr | case | each

        self.require_indent()?;

        let mut children = Vec::new();

        loop {
            let child = match self.lexer.peek() {
                Token::Identifier(_) => match self.expression_definition() {
                    Ok(node) => node,
                    Err(_) => {
                        self.lexer.backtrack();
                        self.expression()?
                    }
                },
                Token::String(_) | Token::Number(_) => self.term()?,
                Token::Keyword(Keyword::Case) => self.case()?,
                Token::Keyword(Keyword::Each) => self.each()?,
                Token::Dedent(_) => break,
                _ => {
                    return Err(ParseError::UnexpectedToken(
                        "block_stmt",
                        self.lexer.peek(),
                        self.lexer.cursor(),
                    ))?;
                }
            };

            children.push(child);

            if self.lexer.peek() == Token::Newline {
                self.lexer.next();
            } else {
                self.require_dedent()?;
                break;
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
                return Err(ParseError::UnexpectedToken(
                    "range",
                    self.lexer.peek(),
                    self.lexer.cursor(),
                ))
            }
        };

        let op = match self.lexer.next() {
            Token::RangeOperator(op) => op,
            _ => {
                return Err(ParseError::UnexpectedToken(
                    "range",
                    self.lexer.peek(),
                    self.lexer.cursor(),
                ))
            }
        };

        let end = match self.lexer.next() {
            Token::Number(num) => Node::Integer(num),
            _ => {
                return Err(ParseError::UnexpectedToken(
                    "range",
                    self.lexer.peek(),
                    self.lexer.cursor(),
                ))
            }
        };

        Ok(Node::Range {
            is_exclusive: op == RangeOperator::Exclusive,
            start: Box::new(start),
            end: Box::new(end),
        })
    }
}
