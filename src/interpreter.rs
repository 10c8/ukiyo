mod stdlib;

use std::{borrow::Borrow, rc::Rc};

use codespan_reporting::diagnostic::{Diagnostic, Label};
use ecow::EcoString;
use rustc_hash::FxHashMap;

use crate::{
    lexer::{Lexer, RangeMode, ToDiagnostic},
    parser::{AstNode, CasePatternKind, Parser},
};
use stdlib::StdLib;

static UNDERLINE: &str = "_";

type IResult = Result<Rc<Value>, InterpreterError>;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Boolean(bool),
    String(EcoString),
    Number(f64),
    List(Rc<[Rc<Value>]>),
    Record(FxHashMap<EcoString, Rc<Value>>),
    Closure(Closure),
    NativeFunction(fn(Vec<Rc<Value>>, Range) -> IResult),
    Any,
    Nil,
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Boolean(value) => write!(f, "{}", value),
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
                    write!(f, "{}: {}", key, value)?;
                }
                write!(f, "}}")
            }
            Value::Closure(_) => write!(f, "closure"),
            Value::NativeFunction(_) => write!(f, "native function"),
            Value::Any => write!(f, "any"),
            Value::Nil => write!(f, "nil"),
        }
    }
}

fn typeof_value(value: &Value) -> &'static str {
    match value {
        Value::Boolean(_) => "boolean",
        Value::String(_) => "string",
        Value::Number(_) => "number",
        Value::List(_) => "list",
        Value::Record(_) => "record",
        Value::Closure(_) => "closure",
        Value::NativeFunction(_) => "native function",
        Value::Any => "any",
        Value::Nil => "nil",
    }
}

#[macro_export]
macro_rules! native_fn {
    ($name:literal; $self:ident: $fn:expr) => {
        $self
            .values
            .insert($name.into(), (Rc::new(Value::NativeFunction($fn)), (0, 0)));
    };
}

#[macro_export]
macro_rules! native_arg_count {
    ($name:literal; $args:ident == $count:expr; $range:expr) => {
        if $args.len() != $count {
            return Err(InterpreterError::NativeFunctionWrongArgumentCount(
                $range,
                $count,
                $args.len(),
            ));
        }
    };
}

#[macro_export]
macro_rules! native_arg {
    ($name:literal; $args:ident @ $index:expr; $range:expr) => {
        if let Some(value) = $args.get($index) {
            value
        } else {
            return Err(InterpreterError::NativeFunctionInvalidArgument(
                $range,
                $index,
                "any".to_string(),
                typeof_value($args.get($index).unwrap()),
            ));
        }
    };
    ($name:literal; $args:ident @ $index:expr => $type:ident @ $range:expr) => {
        if let Some(value) = $args.get($index) {
            match value.borrow() {
                Value::$type(value) => value,
                _ => {
                    return Err(InterpreterError::NativeFunctionInvalidArgument(
                        $range,
                        $index,
                        stringify!($type).to_lowercase(),
                        typeof_value($args.get($index).unwrap()),
                    ))
                }
            }
        } else {
            return Err(InterpreterError::NativeFunctionInvalidArgument(
                $range,
                $index,
                stringify!($type).to_lowercase(),
                typeof_value($args.get($index).unwrap()),
            ));
        }
    };
}

#[derive(Debug, Clone, PartialEq)]
pub struct Environment {
    values: FxHashMap<EcoString, (Rc<Value>, Range)>,
}

impl Environment {
    pub fn new() -> Self {
        Self {
            values: FxHashMap::default(),
        }
    }

    pub fn load_stdlib(&mut self) {
        StdLib::load(self);
    }

    pub fn get(&self, name: &str) -> Option<&(Rc<Value>, Range)> {
        self.values.get(name)
    }

    pub fn set(&mut self, name: &str, value: Rc<Value>, range: Range) {
        self.values.insert(name.into(), (value, range));
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Closure {
    params: Vec<EcoString>,
    body: AstNode,
    range: Range,
    env: Environment,
}

impl Closure {
    pub fn call(&self, args: Vec<Rc<Value>>, caller_range: Option<Range>) -> IResult {
        if self.params.len() != args.len() {
            let caller_range = caller_range.unwrap_or(self.range);
            let callee_range = if caller_range == self.range {
                None
            } else {
                Some(self.range)
            };

            return Err(InterpreterError::WrongArgumentCount(
                caller_range,
                callee_range,
                self.params.len(),
                args.len(),
            ));
        }

        let mut env = self.env.clone();
        for (name, arg) in self.params.iter().zip(args.iter()) {
            env.set(name, arg.clone(), self.body.range());
        }

        self.body.eval(&mut env)
    }
}

type Range = (usize, usize);

#[derive(Debug, Clone)]
pub enum InterpreterError {
    UndefinedVariable(EcoString, Range),
    NotCallable(AstNode, Range),
    NotIterable(Rc<Value>, Range),
    IndexOutOfBounds(&'static str, Range, i64, usize),
    InvalidKey(EcoString, Range),
    WrongArgumentCount(Range, Option<Range>, usize, usize),
    TypeMismatch(Range, &'static str, &'static str),
    FunctionRedeclaration(EcoString, Option<Range>, Range),
    RecordKeyRedeclaration(EcoString, Range),
    InvalidFunctionName(Range),
    InvalidFunctionParam(Range),
    InvalidRecordKey(Range),
    NativeFunctionError(&'static str, Range),
    NativeFunctionWrongArgumentCount(Range, usize, usize),
    NativeFunctionInvalidArgument(Range, usize, String, &'static str),
    StringFormatError(
        usize,
        Option<Diagnostic<usize>>,
        Option<Diagnostic<usize>>,
        Option<Diagnostic<usize>>,
    ),
}

impl ToDiagnostic for InterpreterError {
    fn to_diagnostic(&self) -> Diagnostic<usize> {
        match self {
            InterpreterError::UndefinedVariable(name, range) => Diagnostic::error()
                .with_code("run")
                .with_message(format!("\"{}\" is not defined.", name))
                .with_labels(vec![Label::primary(0, range.0..range.1)]),
            InterpreterError::InvalidKey(key, range) => Diagnostic::error()
                .with_code("run")
                .with_message(format!("\"{}\" is not a valid key.", key))
                .with_labels(vec![Label::primary(0, range.0..range.1)]),
            InterpreterError::NotCallable(value, range) => {
                let typename = match value {
                    AstNode::Case { .. } => "case expression",
                    AstNode::StrLit { .. } => "string",
                    AstNode::NumLit { .. } => "number",
                    AstNode::List { .. } => "list",
                    AstNode::Record { .. } => "record",
                    _ => unreachable!("not callable: {:#?}", value),
                };

                Diagnostic::error()
                    .with_code("run")
                    .with_message(format!("Cannot call a {}.", typename))
                    .with_labels(vec![Label::primary(0, range.0..range.1)])
            }
            InterpreterError::NotIterable(value, range) => Diagnostic::error()
                .with_code("run")
                .with_message(format!("Type `{}` is not iterable.", typeof_value(&value)))
                .with_labels(vec![Label::primary(0, range.0..range.1)]),
            InterpreterError::IndexOutOfBounds(typename, range, index, len) => Diagnostic::error()
                .with_code("run")
                .with_message("Index out of bounds.")
                .with_labels(vec![Label::primary(0, range.0..range.1)])
                .with_notes(vec![format!(
                    "Index {} is out of bounds for {} of length {}.",
                    typename, index, len
                )]),
            InterpreterError::WrongArgumentCount(range, callee_range, expected, got) => {
                let mut labels = vec![Label::primary(0, range.0..range.1)];

                if let Some(callee_range) = callee_range {
                    labels.push(
                        Label::secondary(0, callee_range.0..callee_range.1)
                            .with_message("defined here"),
                    );
                }

                Diagnostic::error()
                    .with_code("run")
                    .with_message("Wrong number of arguments.")
                    .with_labels(labels)
                    .with_notes(vec![format!(
                        "Expected {} {}, got {}.",
                        expected,
                        if *expected == 1 {
                            "argument"
                        } else {
                            "arguments"
                        },
                        got
                    )])
            }
            InterpreterError::TypeMismatch(range, expected, got) => Diagnostic::error()
                .with_code("run")
                .with_message("Type mismatch.")
                .with_labels(vec![Label::primary(0, range.0..range.1)])
                .with_notes(vec![format!("Expected a {}, got a {}.", expected, got)]),
            InterpreterError::FunctionRedeclaration(name, org_range, new_range) => {
                let mut labels = Vec::new();
                labels.push(Label::primary(0, new_range.0..new_range.1));

                if let Some(org_range) = org_range {
                    labels.push(
                        Label::secondary(0, org_range.0..org_range.1)
                            .with_message("first defined here"),
                    );
                }

                let diagnostic = Diagnostic::error()
                    .with_code("run")
                    .with_message(format!("\"{}\" is already defined.", name))
                    .with_labels(labels);

                if org_range.is_none() {
                    diagnostic
                        .with_notes(vec!["You cannot redefine a built-in function.".to_string()])
                } else {
                    diagnostic
                }
            }
            InterpreterError::RecordKeyRedeclaration(name, range) => Diagnostic::error()
                .with_code("run")
                .with_message(format!("Key \"{}\" is already defined.", name))
                .with_labels(vec![Label::primary(0, range.0..range.1)]),
            InterpreterError::InvalidFunctionName(range) => Diagnostic::error()
                .with_code("run")
                .with_message("Invalid function name.")
                .with_labels(vec![Label::primary(0, range.0..range.1)])
                .with_notes(vec![format!("A function cannot be called `_`.")]),
            InterpreterError::InvalidFunctionParam(range) => Diagnostic::error()
                .with_code("run")
                .with_message("Invalid parameter name.")
                .with_labels(vec![Label::primary(0, range.0..range.1)])
                .with_notes(vec![format!("A parameter cannot be called `_`.")]),
            InterpreterError::InvalidRecordKey(range) => Diagnostic::error()
                .with_code("run")
                .with_message("Invalid record key.")
                .with_labels(vec![Label::primary(0, range.0..range.1)])
                .with_notes(vec![format!("A record key cannot be called `_`.")]),
            InterpreterError::NativeFunctionError(message, range) => Diagnostic::error()
                .with_code("run")
                .with_message(*message)
                .with_labels(vec![Label::primary(0, range.0..range.1)]),
            InterpreterError::NativeFunctionWrongArgumentCount(range, expected, got) => {
                Diagnostic::error()
                    .with_code("run")
                    .with_message("Wrong number of arguments.")
                    .with_labels(vec![Label::primary(0, range.0..range.1)])
                    .with_notes(vec![format!(
                        "Expected {} {}, got {}.",
                        expected,
                        if *expected == 1 {
                            "argument"
                        } else {
                            "arguments"
                        },
                        got
                    )])
            }
            InterpreterError::NativeFunctionInvalidArgument(range, index, expected, got) => {
                Diagnostic::error()
                    .with_code("run")
                    .with_message(format!("Invalid argument at index {}.", index))
                    .with_labels(vec![Label::primary(0, range.0..range.1)])
                    .with_notes(vec![format!("Expected a {}, got a {}.", expected, got)])
            }
            InterpreterError::StringFormatError(start, lex_diag, parse_diag, eval_diag) => {
                let mut code = String::new();
                let mut message = EcoString::new();
                let mut labels = Vec::new();
                let mut notes = Vec::new();

                if let Some(lex_diag) = lex_diag {
                    code = lex_diag.code.clone().unwrap();
                    message.push_str(&lex_diag.message);
                    notes = lex_diag.notes.clone();

                    for label in lex_diag.labels.iter() {
                        let mut label = label.clone();
                        label.range.start += *start;
                        label.range.end += *start;

                        labels.push(label);
                    }
                }

                if let Some(parse_diag) = parse_diag {
                    code = parse_diag.code.clone().unwrap();
                    message.push_str(&parse_diag.message);
                    notes = parse_diag.notes.clone();

                    for label in parse_diag.labels.iter() {
                        let mut label = label.clone();
                        label.range.start += *start;
                        label.range.end += *start;

                        labels.push(label);
                    }
                }

                if let Some(eval_diag) = eval_diag {
                    code = eval_diag.code.clone().unwrap();
                    message.push_str(&eval_diag.message);
                    notes = eval_diag.notes.clone();

                    for label in eval_diag.labels.iter() {
                        let mut label = label.clone();
                        label.range.start += *start;
                        label.range.end += *start;

                        labels.push(label);
                    }
                }

                Diagnostic::error()
                    .with_code(code)
                    .with_message(message)
                    .with_labels(labels)
                    .with_notes(notes)
            }
        }
    }
}

impl AstNode {
    pub fn eval(&self, environment: &mut Environment) -> IResult {
        match self {
            AstNode::Prog { body, .. } | AstNode::Block { body, .. } => {
                let mut result = Rc::new(Value::Nil);
                for node in body {
                    result = node.eval(environment)?;
                }
                Ok(result)
            }
            AstNode::ExprStmt { expr, .. } => expr.eval(environment),
            AstNode::FuncDecl {
                id: id_node,
                is_const,
                params: param_list,
                body,
                ..
            } => {
                let (id, id_range) = match id_node.borrow() {
                    AstNode::Ident { name, .. } => {
                        if *name == UNDERLINE {
                            return Err(InterpreterError::InvalidFunctionName(id_node.range()));
                        } else {
                            if let Some((_, range)) = environment.get(name) {
                                return Err(InterpreterError::FunctionRedeclaration(
                                    name.clone(),
                                    Some(*range),
                                    id_node.range(),
                                ));
                            }

                            (name, id_node.range())
                        }
                    }
                    _ => unreachable!(),
                };

                let mut params = Vec::new();
                for param in param_list {
                    match param {
                        AstNode::Ident { name, .. } => {
                            if *name == UNDERLINE {
                                return Err(InterpreterError::InvalidFunctionParam(param.range()));
                            } else {
                                params.push(name.clone())
                            }
                        }
                        _ => unreachable!(),
                    }
                }

                let closure = Closure {
                    params,
                    body: *body.clone(),
                    range: id_range,
                    env: environment.clone(),
                };

                let value = if *is_const {
                    // TODO: Make it lazy
                    closure.call(vec![], None)?
                } else {
                    Rc::new(Value::Closure(closure))
                };

                environment.set(id, value, id_node.as_ref().range());

                Ok(Rc::new(Value::Nil))
            }
            AstNode::FuncRef { id: id_node, .. } => {
                let id = match id_node.as_ref() {
                    AstNode::Ident { name, .. } => name,
                    _ => unreachable!(),
                };

                if let Some((value, _)) = environment.get(id) {
                    Ok(value.clone())
                } else {
                    return Err(InterpreterError::UndefinedVariable(
                        id.clone(),
                        id_node.range(),
                    ));
                }
            }
            AstNode::Lambda {
                params: param_list,
                body,
                range,
            } => {
                let mut params = Vec::new();
                for param in param_list {
                    match param {
                        AstNode::Ident { name, .. } => {
                            if *name == UNDERLINE {
                                return Err(InterpreterError::InvalidFunctionParam(param.range()));
                            } else {
                                params.push(name.clone())
                            }
                        }
                        _ => unreachable!(),
                    }
                }

                let closure = Closure {
                    params,
                    body: *body.clone(),
                    range: *range,
                    env: environment.clone(),
                };

                Ok(Rc::new(Value::Closure(closure)))
            }
            AstNode::Case { expr, cases, .. } => {
                let case_expr = expr.eval(environment)?;

                for case in cases {
                    let (pattern, body) = match case {
                        AstNode::CaseBranch { pattern, body, .. } => (pattern, body),
                        _ => unreachable!(),
                    };

                    match pattern.as_ref() {
                        AstNode::CasePattern { kind, expr, .. } => match kind {
                            CasePatternKind::Any => return Ok(body.eval(environment)?),
                            CasePatternKind::Ident => {
                                let test_name = match expr.as_ref() {
                                    AstNode::Ident { name, .. } => name,
                                    _ => unreachable!(),
                                };

                                let closure = Closure {
                                    params: vec![test_name.clone()],
                                    body: *body.clone(),
                                    range: body.range(),
                                    env: environment.clone(),
                                };

                                let result = closure.call(vec![case_expr], None)?;
                                return Ok(result);
                            }
                            CasePatternKind::Literal => {
                                let test_expr = expr.eval(environment)?;

                                if case_expr == test_expr {
                                    return Ok(body.eval(environment)?);
                                } else {
                                    continue;
                                }
                            } // CasePatternKind::Or => todo!("case: or"),
                        },
                        _ => unreachable!(),
                    }
                }

                Ok(Rc::new(Value::Nil))
            }
            AstNode::FuncCall {
                callee,
                args: arg_list,
                ..
            } => {
                let mut args = Vec::new();
                for arg in arg_list {
                    let value = arg.eval(environment)?;
                    args.push(value);
                }

                match callee.borrow() {
                    AstNode::Ident { .. } | AstNode::FuncRef { .. } => {
                        callee.call(args, environment)
                    }
                    AstNode::FuncCall { .. } => {
                        let closure = callee.eval(environment)?;
                        match closure.borrow() {
                            Value::Closure(closure) => {
                                let result = closure.call(args, Some(self.range()))?;
                                return Ok(result);
                            }
                            _ => unreachable!(),
                        }
                    }
                    _ => {
                        return Err(InterpreterError::NotCallable(
                            *callee.clone(),
                            callee.range(),
                        ));
                    }
                }
            }
            AstNode::Indexing {
                expr: expr_node,
                index: index_node,
                ..
            } => {
                let expr = expr_node.eval(environment)?;
                let index = index_node.eval(environment)?;

                match expr.borrow() {
                    Value::String(value) => {
                        let index = match index.borrow() {
                            Value::Number(value) => *value as i64,
                            value => Err(InterpreterError::TypeMismatch(
                                index_node.range(),
                                "number",
                                typeof_value(&value),
                            ))?,
                        };

                        if index >= value.len() as i64 {
                            return Err(InterpreterError::IndexOutOfBounds(
                                "string",
                                index_node.range(),
                                index,
                                value.len(),
                            ));
                        }

                        let value = if index < 0 {
                            Rc::new(Value::String(
                                value
                                    .chars()
                                    .nth(value.len() - index as usize)
                                    .unwrap()
                                    .into(),
                            ))
                        } else {
                            Rc::new(Value::String(
                                value.chars().nth(index as usize).unwrap().into(),
                            ))
                        };

                        Ok(value)
                    }
                    Value::List(items) => {
                        let index = match index.borrow() {
                            Value::Number(value) => *value as i64,
                            value => Err(InterpreterError::TypeMismatch(
                                index_node.range(),
                                "number",
                                typeof_value(&value),
                            ))?,
                        };

                        if index >= items.len() as i64 {
                            return Err(InterpreterError::IndexOutOfBounds(
                                "list",
                                index_node.range(),
                                index,
                                items.len(),
                            ));
                        }

                        let value = if index < 0 {
                            items[items.len() - index as usize].clone()
                        } else {
                            items[index as usize].clone()
                        };

                        Ok(value)
                    }
                    Value::Record(record) => {
                        let key = match index.borrow() {
                            Value::String(value) => value.clone(),
                            value => Err(InterpreterError::TypeMismatch(
                                index_node.range(),
                                "string",
                                typeof_value(&value),
                            ))?,
                        };

                        if let Some(value) = record.get(&key) {
                            Ok(value.clone())
                        } else {
                            Err(InterpreterError::InvalidKey(key, index_node.range()))?
                        }
                    }
                    _ => {
                        return Err(InterpreterError::NotIterable(
                            expr.clone(),
                            expr_node.range(),
                        ));
                    }
                }
            }
            AstNode::IterationOp {
                id,
                expr: expr_node,
                body,
                ..
            } => {
                let id = match id.as_ref() {
                    AstNode::Ident { name, .. } => name,
                    _ => unreachable!(),
                };

                let env = environment.clone();
                let closure = Closure {
                    params: vec![id.clone()],
                    body: body.as_ref().clone(),
                    range: body.range(),
                    env,
                };

                let expr = expr_node.eval(environment)?;

                match expr.borrow() {
                    Value::List(items) => {
                        let mut result = Vec::new();
                        for item in items.iter() {
                            let value = closure.call(vec![item.clone()], None)?;
                            result.push(value);
                        }

                        Ok(Value::List(result.into()).into())
                    }
                    Value::String(value) => {
                        let mut result = Vec::new();
                        for item in value.chars() {
                            let value =
                                closure.call(vec![Rc::new(Value::String(item.into()))], None)?;
                            result.push(value);
                        }

                        Ok(Value::List(result.into()).into())
                    }
                    Value::Record(_record) => todo!("iterate over records"), // TODO: iterate over records
                    _ => Err(InterpreterError::NotIterable(
                        expr.clone(),
                        expr_node.range(),
                    ))?,
                }
            }
            AstNode::Ident { .. } => self.call(vec![], environment),
            AstNode::BoolLit { value, .. } => Ok(Rc::new(Value::Boolean(*value))),
            AstNode::StrLit {
                value,
                is_format,
                range,
            } => {
                if *is_format {
                    let result = eval_format_string(value, environment, *range)?;
                    Ok(Rc::new(result))
                } else {
                    Ok(Rc::new(Value::String(value.clone().into())))
                }
            }
            AstNode::NumLit { value, .. } => Ok(Rc::new(Value::Number(*value))),
            AstNode::Regex { .. } => todo!("regex"), // TODO: eval regex
            AstNode::List {
                items: item_list, ..
            } => {
                let mut items = Vec::new();
                for item in item_list {
                    let value = item.eval(environment)?;
                    items.push(value);
                }

                Ok(Rc::new(Value::List(items.into())))
            }
            AstNode::Record { keys, values, .. } => {
                let mut record = FxHashMap::default();
                for (key, value) in keys.iter().zip(values.iter()) {
                    let key = match key {
                        AstNode::Ident { name, .. } => {
                            if *name == UNDERLINE {
                                return Err(InterpreterError::InvalidRecordKey(key.range()));
                            } else if record.contains_key(name) {
                                return Err(InterpreterError::RecordKeyRedeclaration(
                                    name.clone(),
                                    key.range(),
                                ));
                            } else {
                                name.clone()
                            }
                        }
                        _ => unreachable!(),
                    };
                    let value = value.eval(environment)?;

                    record.insert(key, value);
                }

                Ok(Rc::new(Value::Record(record)))
            }
            AstNode::Range {
                mode,
                start: start_node,
                end: end_node,
                ..
            } => {
                let start = start_node.eval(environment)?;
                let start = match start.borrow() {
                    Value::Number(value) => *value as usize,
                    value => Err(InterpreterError::TypeMismatch(
                        start_node.range(),
                        "number",
                        typeof_value(&value),
                    ))?,
                };

                let end = end_node.eval(environment)?;
                let end = match end.borrow() {
                    Value::Number(value) => *value as usize,
                    value => Err(InterpreterError::TypeMismatch(
                        end_node.range(),
                        "number",
                        typeof_value(&value),
                    ))?,
                };

                // TODO: Make it lazy
                let mut result = Vec::new();

                match mode {
                    RangeMode::Inclusive => {
                        for i in start..=end {
                            result.push(Rc::new(Value::Number(i as f64)));
                        }
                    }
                    RangeMode::Exclusive => {
                        for i in start..end {
                            result.push(Rc::new(Value::Number(i as f64)));
                        }
                    }
                }

                Ok(Rc::new(Value::List(result.into())))
            }
            _ => unreachable!("eval: {:#?}", self),
        }
    }

    pub fn call(&self, args: Vec<Rc<Value>>, environment: &mut Environment) -> IResult {
        match self {
            AstNode::Ident { name, range } => {
                if *name == UNDERLINE {
                    if args.len() != 0 {
                        return Err(InterpreterError::NotCallable(self.clone(), *range));
                    }

                    return Ok(Value::Any.into());
                }

                if let Some((value, _)) = environment.get(name) {
                    match value.borrow() {
                        Value::Closure(closure) => {
                            let result = closure.call(args, Some(*range))?;
                            return Ok(result);
                        }
                        Value::NativeFunction(method) => {
                            let result = method(args, *range)?;
                            return Ok(result);
                        }
                        _ => {
                            if args.len() != 0 {
                                return Err(InterpreterError::NotCallable(self.clone(), *range));
                            }
                        }
                    }

                    return Ok(value.clone());
                } else {
                    return Err(InterpreterError::UndefinedVariable(name.clone(), *range));
                }
            }
            AstNode::FuncRef { id: id_node, range } => {
                let id = match id_node.as_ref() {
                    AstNode::Ident { name, .. } => name,
                    _ => unreachable!(),
                };

                if let Some((value, _)) = environment.get(id) {
                    match value.borrow() {
                        Value::Closure(closure) => {
                            let result = closure.call(args, Some(*range))?;
                            return Ok(result);
                        }
                        Value::NativeFunction(method) => {
                            let result = method(args, id_node.range())?;
                            return Ok(result);
                        }
                        _ => {
                            if args.len() != 0 {
                                return Err(InterpreterError::NotCallable(
                                    self.clone(),
                                    id_node.range(),
                                ));
                            }
                        }
                    }

                    return Ok(value.clone());
                } else {
                    return Err(InterpreterError::UndefinedVariable(
                        id.clone(),
                        id_node.range(),
                    ));
                }
            }
            _ => todo!("call: {:#?}", self),
        }
    }
}

fn eval_format_string(
    value: &EcoString,
    environment: &mut Environment,
    range: Range,
) -> Result<Value, InterpreterError> {
    let mut result = EcoString::new();

    let mut capture = EcoString::new();
    let mut capture_start = 0;
    let mut is_capturing = false;

    let mut chars = value.chars().enumerate().peekable();

    while let Some((_, c)) = chars.next() {
        if is_capturing {
            match c {
                '}' => {
                    if let Some((_, '}')) = chars.peek() {
                        chars.next();

                        is_capturing = false;

                        let mut lexer = Lexer::new(&capture);
                        if let Err(err) = lexer.lex() {
                            return Err(InterpreterError::StringFormatError(
                                range.0 + capture_start,
                                Some(err.to_diagnostic()),
                                None,
                                None,
                            ));
                        }

                        let mut parser = Parser::new(lexer);
                        let ast = parser.parse();
                        if let Err(err) = ast {
                            return Err(InterpreterError::StringFormatError(
                                range.0 + capture_start,
                                None,
                                Some(parser.error_to_diagnostic(err)),
                                None,
                            ));
                        }

                        let value = ast.unwrap().eval(environment);
                        if let Err(err) = value {
                            return Err(InterpreterError::StringFormatError(
                                range.0 + capture_start,
                                None,
                                None,
                                Some(err.to_diagnostic()),
                            ));
                        }

                        let value = match value.unwrap().borrow() {
                            Value::String(value) => value.clone(),
                            value => EcoString::from(value.to_string()),
                        };

                        result.push_str(&value);
                        capture.clear();
                    } else {
                        capture.push(c);
                    }
                }
                _ => {
                    capture.push(c);
                }
            }
        } else {
            match c {
                '{' => {
                    if let Some((i, '{')) = chars.peek() {
                        is_capturing = true;
                        capture_start = i + 2;

                        chars.next();
                    } else {
                        result.push(c);
                    }
                }
                _ => {
                    result.push(c);
                }
            }
        }
    }

    Ok(Value::String(result.into()))
}
