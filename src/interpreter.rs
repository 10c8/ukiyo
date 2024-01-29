mod stdlib;

use std::{
    borrow::Borrow,
    sync::{Arc, Mutex},
};

use codespan_reporting::diagnostic::{Diagnostic, Label};
use ecow::EcoString;
use once_cell::sync::Lazy;
use rustc_hash::FxHashMap;

use crate::{
    lexer::{Lexer, RangeMode, ToDiagnostic},
    parser::{AstNode, CasePatternKind, Parser},
};
use stdlib::StdLib;

static UNDERLINE: &str = "_";

type IResult = Result<Value, InterpreterError>;

type Value = Arc<ValueData>;

#[derive(Debug, Clone, PartialEq)]
pub enum ValueData {
    Boolean(bool),
    String(EcoString),
    Number(f64),
    List(Arc<[Value]>),
    Record(FxHashMap<EcoString, Value>),
    Closure(Arc<Closure>),
    Partial(Arc<Closure>, Vec<Value>),
    // TODO: Partial application of native functions
    NativeFunction(fn(Vec<Value>, Range) -> IResult),
    Any,
    Nil,
}

impl std::fmt::Display for ValueData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValueData::Boolean(value) => write!(f, "{}", value),
            ValueData::String(value) => write!(f, "\"{}\"", value),
            ValueData::Number(value) => write!(f, "{}", value),
            ValueData::List(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i != 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            ValueData::Record(record) => {
                write!(f, "{{")?;
                for (i, (key, value)) in record.iter().enumerate() {
                    if i != 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", key, value)?;
                }
                write!(f, "}}")
            }
            ValueData::Closure(_) => write!(f, "closure"),
            ValueData::Partial(_, _) => write!(f, "closure"),
            ValueData::NativeFunction(_) => write!(f, "native function"),
            ValueData::Any => write!(f, "any"),
            ValueData::Nil => write!(f, "nil"),
        }
    }
}

fn typeof_value(value: &ValueData) -> &'static str {
    match value {
        ValueData::Boolean(_) => "boolean",
        ValueData::String(_) => "string",
        ValueData::Number(_) => "number",
        ValueData::List(_) => "list",
        ValueData::Record(_) => "record",
        ValueData::Closure(_) => "closure",
        ValueData::Partial(_, _) => "closure",
        ValueData::NativeFunction(_) => "native function",
        ValueData::Any => "any",
        ValueData::Nil => "nil",
    }
}

#[macro_export]
macro_rules! native_fn {
    ($name:literal; $self:ident: $fn:expr) => {
        $self.store.insert(
            $name.into(),
            (Arc::new(ValueData::NativeFunction($fn)), (0, 0)),
        );
    };
}

#[macro_export]
macro_rules! native_arg_count {
    ($name:literal; $args:ident == $count:expr; $range:expr) => {
        if $args.len() != $count {
            return Err(InterpreterError::NativeFunctionWrongArgumentCount(
                $name,
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
                ValueData::$type(value) => value,
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

#[derive(Debug, Clone)]
pub struct Environment {
    store: FxHashMap<EcoString, (Value, Range)>,
    parent: Option<Arc<Mutex<Environment>>>,
}

impl Environment {
    pub fn new() -> Self {
        Self {
            store: FxHashMap::default(),
            parent: None,
        }
    }

    pub fn new_child(parent: Arc<Mutex<Self>>) -> Self {
        Self {
            store: FxHashMap::default(),
            parent: Some(parent),
        }
    }

    pub fn load_stdlib(&mut self) {
        StdLib::load(self);
    }

    pub fn get(&self, name: &str) -> Option<(Value, Range)> {
        match self.store.get(name) {
            Some(value) => Some(value.clone()),
            None => match &self.parent {
                Some(parent) => parent.lock().unwrap().get(name),
                None => None,
            },
        }
    }

    pub fn set(&mut self, name: &str, value: Value, range: Range) {
        self.store.insert(name.into(), (value, range));
    }
}

static GLOBAL_ENV: Lazy<Environment> = Lazy::new(|| {
    let mut env = Environment::new();
    env.load_stdlib();
    env
});

#[derive(Debug, Clone)]
pub struct Closure {
    id: Option<EcoString>,
    params: Vec<EcoString>,
    body: AstNode,
    range: Range,
    env: Arc<Mutex<Environment>>,
}

impl PartialEq for Closure {
    fn eq(&self, other: &Self) -> bool {
        let env = self.env.lock().unwrap();
        let other_env = other.env.lock().unwrap();

        self.id == other.id
            && self.params == other.params
            && self.body == other.body
            && self.range == other.range
            && env.store == other_env.store
    }
}

impl Closure {
    pub fn anon_call(&self, args: Vec<Value>, caller_range: Option<Range>) -> IResult {
        if args.len() != self.params.len() {
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

        let mut environment = Arc::new(Mutex::new(Environment::new_child(self.env.clone())));

        {
            let mut environment = environment.lock().unwrap();
            for (name, arg) in self.params.iter().zip(args.iter()) {
                if *name == UNDERLINE {
                    continue;
                }

                environment.set(name, arg.clone(), self.body.range());
            }
        }

        self.body.eval(&mut environment)
    }

    pub fn call(self: &Arc<Self>, args: Vec<Value>, caller_range: Option<Range>) -> IResult {
        let caller_range = caller_range.unwrap_or(self.range);
        let callee_range = if caller_range == self.range {
            None
        } else {
            Some(self.range)
        };

        if args.len() > self.params.len() {
            return Err(InterpreterError::WrongArgumentCount(
                caller_range,
                callee_range,
                self.params.len(),
                args.len(),
            ));
        }

        if args.len() < self.params.len() {
            if args.is_empty() {
                return Err(InterpreterError::WrongArgumentCount(
                    caller_range,
                    callee_range,
                    self.params.len(),
                    0,
                ));
            }

            return Ok(Arc::new(ValueData::Partial(self.clone(), args)));
        }

        let mut environment = Arc::new(Mutex::new(Environment::new_child(self.env.clone())));

        {
            let mut environment = environment.lock().unwrap();

            for (name, arg) in self.params.iter().zip(args.iter()) {
                if *name == UNDERLINE {
                    continue;
                }

                environment.set(name, arg.clone(), self.body.range());
            }
        }

        self.body.eval(&mut environment)
    }
}

type Range = (usize, usize);

#[derive(Debug, Clone)]
pub enum InterpreterError {
    UndefinedVariable(EcoString, Range),
    NotCallable(AstNode, Range),
    NotIterable(Value, Range),
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
    NativeFunctionWrongArgumentCount(&'static str, Range, usize, usize),
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
            InterpreterError::NativeFunctionWrongArgumentCount(name, range, expected, got) => {
                Diagnostic::error()
                    .with_code("run")
                    .with_message(format!("{}: Wrong number of arguments.", name))
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

fn env_get(env: &Arc<Mutex<Environment>>, name: &str) -> Option<(Value, Range)> {
    if let Some((value, range)) = env.lock().unwrap().get(name) {
        Some((value, range))
    } else {
        GLOBAL_ENV.get(name)
    }
}

impl AstNode {
    pub fn eval(&self, environment: &mut Arc<Mutex<Environment>>) -> IResult {
        match self {
            AstNode::Prog { body, .. } | AstNode::Block { body, .. } => {
                let mut result = Arc::new(ValueData::Nil);
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
                            if let Some((_, range)) = env_get(environment, name) {
                                let range = if range == (0, 0) { None } else { Some(range) };

                                return Err(InterpreterError::FunctionRedeclaration(
                                    name.clone(),
                                    range,
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

                let env = Arc::new(Mutex::new(Environment::new_child(environment.clone())));
                let closure = Closure {
                    id: if !is_const { Some(id.clone()) } else { None },
                    params,
                    body: *body.clone(),
                    range: id_range,
                    env,
                };

                let value = if *is_const {
                    // TODO: Make it lazy
                    closure.anon_call(vec![], None)?
                } else {
                    Arc::new(ValueData::Closure(Arc::new(closure)))
                };

                environment
                    .lock()
                    .unwrap()
                    .set(id, value, id_node.as_ref().range());

                Ok(Arc::new(ValueData::Nil))
            }
            AstNode::FuncRef { id: id_node, .. } => {
                let id = match id_node.as_ref() {
                    AstNode::Ident { name, .. } => name,
                    _ => unreachable!(),
                };

                if let Some((value, _)) = env_get(environment, id) {
                    Ok(value)
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
                        AstNode::Ident { name, .. } => params.push(name.clone()),
                        _ => unreachable!(),
                    }
                }

                let env = Arc::new(Mutex::new(Environment::new_child(environment.clone())));
                let closure = Closure {
                    id: None,
                    params,
                    body: *body.clone(),
                    range: *range,
                    env,
                };

                Ok(Arc::new(ValueData::Closure(Arc::new(closure))))
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

                                let env = Arc::new(Mutex::new(Environment::new_child(
                                    environment.clone(),
                                )));
                                let closure = Closure {
                                    id: None,
                                    params: vec![test_name.clone()],
                                    body: *body.clone(),
                                    range: body.range(),
                                    env,
                                };

                                let result = closure.anon_call(vec![case_expr], None)?;
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

                Ok(Arc::new(ValueData::Nil))
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
                    AstNode::FuncCall { .. } | AstNode::Lambda { .. } => {
                        let closure = callee.eval(environment)?;
                        match closure.borrow() {
                            ValueData::Closure(closure) => {
                                let result = closure.call(args, Some(self.range()))?;
                                return Ok(result);
                            }
                            ValueData::Partial(closure, partial_args) => {
                                let mut new_args = partial_args.clone();
                                new_args.extend(args);

                                let result = closure.call(new_args, Some(self.range()))?;
                                return Ok(result);
                            }
                            _ => {
                                return Err(InterpreterError::NotCallable(
                                    *callee.clone(),
                                    callee.range(),
                                ));
                            }
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
            AstNode::IndexingOp {
                expr: expr_node,
                index: index_node,
                ..
            } => {
                let expr = expr_node.eval(environment)?;
                let index = index_node.eval(environment)?;

                match expr.borrow() {
                    ValueData::String(value) => {
                        let index = match index.borrow() {
                            ValueData::Number(value) => *value as i64,
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
                            Arc::new(ValueData::String(
                                value
                                    .chars()
                                    .nth(value.len() - index as usize)
                                    .unwrap()
                                    .into(),
                            ))
                        } else {
                            Arc::new(ValueData::String(
                                value.chars().nth(index as usize).unwrap().into(),
                            ))
                        };

                        Ok(value)
                    }
                    ValueData::List(items) => {
                        let index = match index.borrow() {
                            ValueData::Number(value) => *value as i64,
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
                    ValueData::Record(record) => {
                        let key = match index.borrow() {
                            ValueData::String(value) => value.clone(),
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
            AstNode::ConcatOp {
                left: left_node,
                right: right_node,
                ..
            } => {
                let left = left_node.eval(environment)?;
                let right = right_node.eval(environment)?;

                if typeof_value(&left) != typeof_value(&right) {
                    return Err(InterpreterError::TypeMismatch(
                        self.range(),
                        typeof_value(&left),
                        typeof_value(&right),
                    ));
                }

                match left.borrow() {
                    ValueData::String(left) => {
                        let right = match right.borrow() {
                            ValueData::String(right) => right,
                            _ => unreachable!(),
                        };

                        let mut result = left.clone();
                        result.push_str(right);

                        Ok(Arc::new(ValueData::String(result)))
                    }
                    ValueData::List(left) => {
                        let right = match right.borrow() {
                            ValueData::List(right) => right,
                            _ => unreachable!(),
                        };

                        let mut result = Vec::new();
                        result.extend_from_slice(left);
                        result.extend_from_slice(right);

                        Ok(Arc::new(ValueData::List(result.into())))
                    }
                    _ => unreachable!(),
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

                let env = Arc::new(Mutex::new(Environment::new_child(environment.clone())));
                let closure = Closure {
                    id: None,
                    params: vec![id.clone()],
                    body: body.as_ref().clone(),
                    range: body.range(),
                    env,
                };

                let expr = expr_node.eval(environment)?;

                match expr.borrow() {
                    ValueData::List(items) => {
                        let mut result = Vec::new();
                        for item in items.iter() {
                            let value = closure.anon_call(vec![item.clone()], None)?;
                            result.push(value);
                        }

                        Ok(ValueData::List(result.into()).into())
                    }
                    ValueData::String(value) => {
                        let mut result = Vec::new();
                        for item in value.chars() {
                            let value = closure
                                .anon_call(vec![Arc::new(ValueData::String(item.into()))], None)?;
                            result.push(value);
                        }

                        Ok(ValueData::List(result.into()).into())
                    }
                    ValueData::Record(_record) => todo!("iterate over records"), // TODO: iterate over records
                    _ => Err(InterpreterError::NotIterable(
                        expr.clone(),
                        expr_node.range(),
                    ))?,
                }
            }
            AstNode::Ident { .. } => self.call(vec![], environment),
            AstNode::BoolLit { value, .. } => Ok(Arc::new(ValueData::Boolean(*value))),
            AstNode::StrLit {
                value,
                is_format,
                range,
            } => {
                if *is_format {
                    let result = eval_format_string(value, environment, *range)?;
                    Ok(result)
                } else {
                    Ok(Arc::new(ValueData::String(value.clone().into())))
                }
            }
            AstNode::NumLit { value, .. } => Ok(Arc::new(ValueData::Number(*value))),
            AstNode::Regex { .. } => todo!("regex"), // TODO: eval regex
            AstNode::List {
                items: item_list, ..
            } => {
                let mut items = Vec::new();
                for item in item_list {
                    let value = item.eval(environment)?;
                    items.push(value);
                }

                Ok(Arc::new(ValueData::List(items.into())))
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

                Ok(Arc::new(ValueData::Record(record)))
            }
            AstNode::Range {
                mode,
                start: start_node,
                end: end_node,
                ..
            } => {
                let start = start_node.eval(environment)?;
                let start = match start.borrow() {
                    ValueData::Number(value) => *value as usize,
                    value => Err(InterpreterError::TypeMismatch(
                        start_node.range(),
                        "number",
                        typeof_value(&value),
                    ))?,
                };

                let end = end_node.eval(environment)?;
                let end = match end.borrow() {
                    ValueData::Number(value) => *value as usize,
                    value => Err(InterpreterError::TypeMismatch(
                        end_node.range(),
                        "number",
                        typeof_value(&value),
                    ))?,
                };

                let is_rev = start > end;

                // TODO: Make it lazy
                let mut result = Vec::new();

                match mode {
                    RangeMode::Inclusive => {
                        let range = std::cmp::min(start, end)..=std::cmp::max(start, end);

                        if is_rev {
                            for i in range.rev() {
                                result.push(Arc::new(ValueData::Number(i as f64)));
                            }
                        } else {
                            for i in range {
                                result.push(Arc::new(ValueData::Number(i as f64)));
                            }
                        }
                    }
                    RangeMode::Exclusive => {
                        let range = std::cmp::min(start, end)..std::cmp::max(start, end);

                        if is_rev {
                            for i in range.rev() {
                                result.push(Arc::new(ValueData::Number(i as f64)));
                            }
                        } else {
                            for i in range {
                                result.push(Arc::new(ValueData::Number(i as f64)));
                            }
                        }
                    }
                }

                Ok(Arc::new(ValueData::List(result.into())))
            }
            _ => unreachable!("eval: {:#?}", self),
        }
    }

    pub fn call(&self, args: Vec<Value>, environment: &mut Arc<Mutex<Environment>>) -> IResult {
        match self {
            AstNode::Ident { name, range } => {
                if *name == UNDERLINE {
                    if args.len() != 0 {
                        return Err(InterpreterError::NotCallable(self.clone(), *range));
                    }

                    return Ok(ValueData::Any.into());
                }

                let value = match env_get(environment, name) {
                    Some((value, _)) => value,
                    None => {
                        return Err(InterpreterError::UndefinedVariable(
                            name.clone(),
                            self.range(),
                        ));
                    }
                };

                match value.borrow() {
                    ValueData::Closure(closure) => {
                        let result = closure.call(args, Some(*range))?;
                        return Ok(result);
                    }
                    ValueData::Partial(closure, partial_args) => {
                        let mut new_args = partial_args.clone();
                        new_args.extend(args);

                        let result = closure.call(new_args, Some(*range))?;
                        return Ok(result);
                    }
                    ValueData::NativeFunction(method) => {
                        let result = method(args, *range)?;
                        return Ok(result);
                    }
                    _ => {
                        if args.len() != 0 {
                            return Err(InterpreterError::NotCallable(self.clone(), *range));
                        }
                    }
                }

                Ok(value)
            }
            AstNode::FuncRef { id: id_node, range } => {
                let id = match id_node.as_ref() {
                    AstNode::Ident { name, .. } => name,
                    _ => unreachable!(),
                };

                let value = match env_get(environment, id) {
                    Some((value, _)) => value,
                    None => {
                        return Err(InterpreterError::UndefinedVariable(
                            id.clone(),
                            id_node.range(),
                        ));
                    }
                };

                match value.borrow() {
                    ValueData::Closure(closure) => {
                        let result = closure.call(args, Some(*range))?;
                        return Ok(result);
                    }
                    ValueData::Partial(closure, partial_args) => {
                        let mut new_args = partial_args.clone();
                        new_args.extend(args);

                        let result = closure.call(new_args, Some(*range))?;
                        return Ok(result);
                    }
                    ValueData::NativeFunction(method) => {
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

                Ok(value)
            }
            _ => todo!("call: {:#?}", self),
        }
    }
}

fn eval_format_string(
    value: &EcoString,
    environment: &Arc<Mutex<Environment>>,
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

                        let mut env =
                            Arc::new(Mutex::new(Environment::new_child(environment.clone())));

                        let value = ast.unwrap().eval(&mut env);
                        if let Err(err) = value {
                            return Err(InterpreterError::StringFormatError(
                                range.0 + capture_start,
                                None,
                                None,
                                Some(err.to_diagnostic()),
                            ));
                        }

                        let value = match value.unwrap().borrow() {
                            ValueData::String(value) => value.clone(),
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

    Ok(Arc::new(ValueData::String(result.into())))
}
