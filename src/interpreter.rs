use std::{collections::HashMap, fmt};

use codespan_reporting::{
    diagnostic::{Diagnostic, Label},
    files::SimpleFiles,
    term::{
        self,
        termcolor::{ColorChoice, StandardStream},
        Config,
    },
};

use crate::{
    lexer::{Lexer, RangeMode},
    parser::{AstNode, CasePatternKind, Parser},
};

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    String(String),
    Number(f64),
    List(Vec<Value>),
    Record(HashMap<String, Value>),
    Closure(Closure),
    Nil,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::String(value) => write!(f, "\"{}\"", value),
            Value::Number(value) => write!(f, "{}", value),
            Value::List(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i != 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            Value::Record(record) => {
                write!(f, "{{")?;
                for (i, (key, value)) in record.iter().enumerate() {
                    if i != 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{} = {}", key, value)?;
                }
                write!(f, "}}")
            }
            Value::Closure(_) => write!(f, "closure"),
            Value::Nil => write!(f, "nil"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Environment {
    values: HashMap<String, Value>,
}

impl Environment {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.values.get(name)
    }

    pub fn set(&mut self, name: &str, value: Value) {
        self.values.insert(name.to_string(), value);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Closure {
    #[allow(dead_code)]
    callee_id: Option<String>,
    param: Option<String>,
    body: AstNode,
    env: Environment,
}

type Range = (usize, usize);

#[derive(Debug, Clone)]
pub enum InterpreterError {
    UndefinedVariable(String, Range),
    NotCallable(AstNode, Range),
}

type IResult = Result<Value, InterpreterError>;

pub struct Interpreter {
    program: AstNode,
    fmt_range: Option<Range>,
}

impl Interpreter {
    pub fn new(program: AstNode) -> Self {
        Self {
            program,
            fmt_range: None,
        }
    }

    pub fn run(&mut self) -> IResult {
        let mut environment = Environment::new();
        self.eval(self.program.clone(), &mut environment)
    }

    pub fn display_error(&mut self, files: SimpleFiles<&str, &str>, error: InterpreterError) {
        let writer = StandardStream::stderr(ColorChoice::Auto);
        let config = Config::default();

        let diagnostic = match error {
            InterpreterError::UndefinedVariable(name, range) => {
                let range = match self.fmt_range {
                    Some(fmt_range) => fmt_range,
                    None => range,
                };

                Diagnostic::error()
                    .with_code("run")
                    .with_message(format!("\"{}\" is not defined.", name))
                    .with_labels(vec![Label::primary(0, range.0..range.1)])
            }
            InterpreterError::NotCallable(value, range) => {
                let range = match self.fmt_range {
                    Some(fmt_range) => fmt_range,
                    None => range,
                };

                let typename = match value {
                    AstNode::Case { .. } => "case expression",
                    AstNode::StrLit { .. } => "string",
                    AstNode::NumLit { .. } => "number",
                    AstNode::List { .. } => "list",
                    AstNode::Record { .. } => "record",
                    _ => unreachable!(),
                };

                Diagnostic::error()
                    .with_code("run")
                    .with_message(format!("Cannot call a {}.", typename))
                    .with_labels(vec![Label::primary(0, range.0..range.1)])
            }
        };

        term::emit(&mut writer.lock(), &config, &files, &diagnostic).unwrap();
    }

    fn apply_closure(
        &mut self,
        callee: Option<AstNode>,
        closure: Closure,
        arg: Option<Value>,
    ) -> IResult {
        if closure.param.is_none() && arg.is_some() {
            let callee = match callee {
                Some(callee) => callee,
                None => unreachable!(),
            };

            return Err(InterpreterError::NotCallable(closure.body, callee.range()));
        }

        let mut env = closure.env;
        if let Some(param) = closure.param {
            if let Some(arg) = arg {
                env.set(&param, arg);
            }
        }

        self.eval(closure.body, &mut env)
    }

    fn eval_fmt_string(
        &mut self,
        value: &String,
        range: Range,
        environment: &mut Environment,
    ) -> IResult {
        let mut result = String::new();
        let mut chars = value.chars().peekable();

        let mut is_inside = false;
        let mut buffer = String::new();

        while let Some(c) = chars.next() {
            if c == '{' {
                is_inside = true;
                continue;
            }

            if c == '}' {
                is_inside = false;

                let mut lexer = Lexer::new(&buffer);
                lexer.lex().expect("fmt: lexer error");

                let mut parser = Parser::new(lexer);
                let ast = parser.parse().expect("fmt: parser error");

                self.fmt_range = Some(range);

                let value = self.eval(ast, environment)?;
                let value = match value {
                    Value::String(value) => value,
                    value => value.to_string(),
                };

                self.fmt_range = None;

                result.push_str(&value);

                buffer.clear();
                continue;
            }

            if is_inside {
                buffer.push(c);
                continue;
            }

            result.push(c);
        }

        Ok(Value::String(result))
    }

    fn eval(&mut self, node: AstNode, environment: &mut Environment) -> IResult {
        let result = match node {
            AstNode::Prog { body, .. } => {
                let mut result = Value::Nil;
                for node in body {
                    result = self.eval(node, environment)?;
                }
                result
            }
            AstNode::ExprStmt { expr, .. } => self.eval(*expr, environment)?,
            AstNode::Block { body, .. } => {
                let mut result = Value::Nil;
                for node in body {
                    result = self.eval(node, environment)?;
                }
                result
            }
            AstNode::FuncDecl {
                id, param, body, ..
            } => {
                let id = match id {
                    Some(id) => match id.as_ref() {
                        AstNode::Ident { name, .. } => Some(name.clone()),
                        _ => unreachable!(),
                    },
                    None => None,
                };

                let param = match param {
                    Some(param) => match *param {
                        AstNode::Ident { name, .. } => Some(name.clone()),
                        _ => unreachable!(),
                    },
                    _ => None,
                };

                let closure = Value::Closure(Closure {
                    callee_id: id.clone(),
                    param,
                    body: body.as_ref().clone(),
                    env: environment.clone(),
                });

                if let Some(id) = id {
                    environment.set(&id, closure.clone());
                }

                closure
            }
            AstNode::Lambda { param, body, .. } => {
                let param = match *param {
                    AstNode::Ident { name, .. } => Some(name.clone()),
                    _ => unreachable!(),
                };

                let closure = Closure {
                    callee_id: None,
                    param,
                    body: body.as_ref().clone(),
                    env: environment.clone(),
                };

                Value::Closure(closure)
            }
            AstNode::Case { expr, cases, .. } => {
                let case_expr = self.eval(*expr, environment)?;

                for case in cases {
                    let (pattern, body) = match case {
                        AstNode::CaseBranch { pattern, body, .. } => (pattern, body),
                        _ => unreachable!(),
                    };

                    match *pattern {
                        AstNode::CasePattern { kind, expr, .. } => match kind {
                            CasePatternKind::Expr => {
                                let expr = self.eval(*expr, environment)?;

                                if case_expr == expr {
                                    return self.eval(*body, environment);
                                }
                            }
                            CasePatternKind::Or => {
                                let list = match self.eval(*expr, environment)? {
                                    Value::List(list) => list,
                                    _ => unreachable!(),
                                };

                                for item in list {
                                    if case_expr == item {
                                        return self.eval(*body, environment);
                                    }
                                }
                            }
                            CasePatternKind::Empty => return self.eval(*body, environment),
                        },
                        _ => unreachable!(),
                    }
                }

                Value::Nil
            }
            AstNode::FuncCall {
                callee,
                arg: first_arg,
                ..
            } => {
                let first_arg = match first_arg {
                    Some(arg) => Some(self.eval(*arg, environment)?),
                    None => None,
                };

                let result = match *callee {
                    AstNode::Ident { ref name, range } => match environment.get(&name) {
                        Some(Value::Closure(closure)) => {
                            self.apply_closure(Some(*callee), closure.clone(), first_arg)?
                        }
                        Some(value) => {
                            if first_arg.is_some() {
                                return Err(InterpreterError::NotCallable(
                                    *callee.clone(),
                                    callee.range(),
                                ));
                            } else {
                                value.clone()
                            }
                        }
                        _ => {
                            return Err(InterpreterError::UndefinedVariable(
                                name.to_string(),
                                range,
                            ))
                        }
                    },
                    AstNode::FuncCall { callee, arg, .. } => {
                        let closure = self.eval(*callee.clone(), environment)?;
                        let closure = match closure {
                            Value::Closure(closure) => closure,
                            _ => {
                                return Err(InterpreterError::NotCallable(
                                    *callee.clone(),
                                    callee.range(),
                                ))
                            }
                        };

                        let arg = match arg {
                            Some(arg) => Some(self.eval(*arg, environment)?),
                            None => None,
                        };

                        let left = self.apply_closure(Some(*callee.clone()), closure, arg)?;
                        match left {
                            Value::Closure(left) => {
                                self.apply_closure(Some(*callee), left, first_arg)?
                            }
                            _ => left,
                        }
                    }
                    _ => {
                        return Err(InterpreterError::NotCallable(
                            *callee.clone(),
                            callee.range(),
                        ))
                    }
                };

                result
            }
            AstNode::IterationOp { id, expr, body, .. } => {
                let id = match *id {
                    AstNode::Ident { name, .. } => name,
                    _ => unreachable!(),
                };

                let expr = self.eval(*expr, environment)?;
                match expr {
                    Value::List(items) => {
                        let mut result = Vec::new();

                        for item in items {
                            let mut env = environment.clone();
                            env.set(&id.clone(), item);

                            let closure = Closure {
                                callee_id: None,
                                param: Some(id.clone()),
                                body: body.as_ref().clone(),
                                env,
                            };

                            result.push(self.apply_closure(None, closure, None)?);
                        }

                        Value::List(result)
                    }
                    _ => panic!("{:#?} is not iterable", expr),
                }
            }
            AstNode::Ident { ref name, .. } => match environment.get(&name) {
                Some(Value::Closure(closure)) => {
                    self.apply_closure(Some(node), closure.clone(), None)?
                }
                Some(value) => value.clone(),
                None => {
                    return Err(InterpreterError::UndefinedVariable(
                        name.to_string(),
                        node.range(),
                    ))
                }
            },
            AstNode::StrLit {
                value,
                is_format,
                range,
            } => {
                if is_format {
                    self.eval_fmt_string(&value, range, environment)?
                } else {
                    Value::String(value.clone())
                }
            }
            AstNode::NumLit { value, .. } => Value::Number(value),
            AstNode::List { items, .. } => {
                let mut result = Vec::new();

                for item in items {
                    let value = self.eval(item, environment)?;
                    result.push(value);
                }

                Value::List(result)
            }
            AstNode::Record { keys, values, .. } => {
                let mut record = HashMap::new();

                for (key, value) in keys.iter().zip(values.iter()) {
                    let key = match key {
                        AstNode::StrLit { value, .. } => value.clone(),
                        _ => unreachable!(),
                    };
                    let value = self.eval(value.clone(), environment)?;

                    record.insert(key, value);
                }

                Value::Record(record)
            }
            AstNode::Range {
                mode, start, end, ..
            } => {
                let start = match self.eval(*start, environment)? {
                    Value::Number(value) => value,
                    _ => panic!("start is not a number"),
                };

                let end = match self.eval(*end, environment)? {
                    Value::Number(value) => value,
                    _ => panic!("end is not a number"),
                };

                let mut result = Vec::new();

                match mode {
                    RangeMode::Inclusive => {
                        for i in start as usize..=end as usize {
                            result.push(Value::Number(i as f64));
                        }
                    }
                    RangeMode::Exclusive => {
                        for i in start as usize..end as usize {
                            result.push(Value::Number(i as f64));
                        }
                    }
                }

                Value::List(result)
            }
            _ => todo!("eval {:?}", node),
        };

        Ok(result)
    }
}
