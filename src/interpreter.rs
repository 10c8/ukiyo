use std::{borrow::Borrow, rc::Rc};

use codespan_reporting::{
    diagnostic::{Diagnostic, Label},
    files::SimpleFiles,
    term::{
        self,
        termcolor::{ColorChoice, StandardStream},
        Config,
    },
};
use ecow::EcoString;
use rand::Rng;
use regex::Regex;
use rustc_hash::FxHashMap;

use crate::{
    lexer::RangeMode,
    llm::{LLMConfig, LLMType, LLM},
    parser::{AstNode, CasePatternKind},
};

static UNDERLINE: &str = "_";

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    String(EcoString),
    Number(f64),
    List(Rc<[Rc<Value>]>),
    Record(FxHashMap<EcoString, Rc<Value>>),
    Closure(Closure),
    NativeFunction(fn(Vec<Rc<Value>>, Range) -> IResult),
    Any,
    Nil,
}

impl Default for Value {
    fn default() -> Self {
        Self::Nil
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
            Value::NativeFunction(_) => write!(f, "native function"),
            Value::Any => write!(f, "any"),
            Value::Nil => write!(f, "nil"),
        }
    }
}

fn typeof_value(value: &Value) -> &'static str {
    match value {
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

macro_rules! native_fn {
    ($name:literal; $self:ident: $fn:expr) => {
        $self
            .values
            .insert($name.into(), (Rc::new(Value::NativeFunction($fn)), (0, 0)));
    };
}

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

macro_rules! native_arg {
    ($name:literal; $args:ident@$index:expr => $type:ident @ $range:expr) => {
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
        // Debug functions
        native_fn!("trace"; self: Self::std_fn_trace);

        // LLM functions
        native_fn!("gen"; self: Self::std_fn_gen);

        // Math functions
        native_fn!("add"; self: Self::std_fn_add);
        native_fn!("sub"; self: Self::std_fn_sub);
        native_fn!("mul"; self: Self::std_fn_mul);
        native_fn!("div"; self: Self::std_fn_div);
        native_fn!("mod"; self: Self::std_fn_mod);
        native_fn!("rand"; self: Self::std_fn_rand);
        native_fn!("randf"; self: Self::std_fn_randf);
        native_fn!("randl"; self: Self::std_fn_randl);

        // List functions
        native_fn!("len"; self: Self::std_fn_len);
        native_fn!("join"; self: Self::std_fn_join);
        native_fn!("init"; self: Self::std_fn_init);
        native_fn!("last"; self: Self::std_fn_last);
        native_fn!("head"; self: Self::std_fn_head);
        native_fn!("tail"; self: Self::std_fn_tail);
    }

    pub fn get(&self, name: &str) -> Option<&(Rc<Value>, Range)> {
        self.values.get(name)
    }

    pub fn set(&mut self, name: &str, value: Rc<Value>, range: Range) {
        self.values.insert(name.into(), (value, range));
    }

    fn std_fn_trace(args: Vec<Rc<Value>>, range: Range) -> IResult {
        native_arg_count!("trace"; args == 1; range);

        let value = native_arg!("trace"; args @ 0 => String @ range);

        println!("[trace] {}", value);

        Ok(Value::Nil.into())
    }

    fn std_fn_gen(args: Vec<Rc<Value>>, range: Range) -> IResult {
        native_arg_count!("gen"; args == 2; range);

        let prompt = native_arg!("gen"; args @ 0 => String @ range);
        let options = native_arg!("gen"; args @ 1 => Record @ range);

        let context = if let Some(context) = options.get("ctx") {
            match context.borrow() {
                Value::String(value) => Some(value.clone()),
                _ => None,
            }
        } else {
            None
        };

        let max_tokens = if let Some(max_tokens) = options.get("max") {
            match max_tokens.borrow() {
                Value::Number(value) => Some(*value as usize),
                _ => None,
            }
        } else {
            None
        };

        let temperature = if let Some(temperature) = options.get("temp") {
            match temperature.borrow() {
                Value::Number(value) => Some(*value as f32),
                _ => None,
            }
        } else {
            None
        };

        let stop = if let Some(stop) = options.get("stop") {
            match stop.borrow() {
                Value::List(items) => {
                    let mut result = Vec::new();

                    for item in items.iter() {
                        match item.borrow() {
                            Value::String(value) => result.push(value.clone()),
                            _ => {
                                return Err(InterpreterError::NativeFunctionError(
                                    "gen: stop must be a list of strings",
                                ));
                            }
                        }
                    }

                    Some(result)
                }
                _ => None,
            }
        } else {
            None
        };

        let config = LLMConfig {
            llm_type: LLMType::OpenAI {
                model: "gpt-3.5-turbo",
                api_key: "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",
                org: "org-UJZWWVz1mXpuSYKHYnchbjHq",
            },
            ..Default::default()
        };
        let llm = LLM::new(config);

        let result = llm.generate(prompt, context, max_tokens, temperature, stop);

        Ok(Value::String(result.into()).into())
    }

    // Stdlib: Math
    fn std_fn_add(args: Vec<Rc<Value>>, range: Range) -> IResult {
        native_arg_count!("add"; args == 2; range);

        let a = native_arg!("add"; args @ 0 => Number @ range);
        let b = native_arg!("add"; args @ 1 => Number @ range);

        let result = a + b;

        Ok(Value::Number((result as i64) as f64).into())
    }

    fn std_fn_sub(args: Vec<Rc<Value>>, range: Range) -> IResult {
        native_arg_count!("sub"; args == 2; range);

        let a = native_arg!("sub"; args @ 0 => Number @ range);
        let b = native_arg!("sub"; args @ 1 => Number @ range);

        let result = a - b;

        Ok(Value::Number((result as i64) as f64).into())
    }

    fn std_fn_mul(args: Vec<Rc<Value>>, range: Range) -> IResult {
        native_arg_count!("mul"; args == 2; range);

        let a = native_arg!("mul"; args @ 0 => Number @ range);
        let b = native_arg!("mul"; args @ 1 => Number @ range);

        let result = a * b;

        Ok(Value::Number((result as i64) as f64).into())
    }

    fn std_fn_div(args: Vec<Rc<Value>>, range: Range) -> IResult {
        native_arg_count!("div"; args == 2; range);

        let a = native_arg!("div"; args @ 0 => Number @ range);
        let b = native_arg!("div"; args @ 1 => Number @ range);

        let result = a / b;

        Ok(Value::Number((result as i64) as f64).into())
    }

    fn std_fn_mod(args: Vec<Rc<Value>>, range: Range) -> IResult {
        native_arg_count!("mod"; args == 2; range);

        let a = native_arg!("mod"; args @ 0 => Number @ range);
        let b = native_arg!("mod"; args @ 1 => Number @ range);

        Ok(Value::Number(a % b).into())
    }

    fn std_fn_rand(args: Vec<Rc<Value>>, range: Range) -> IResult {
        native_arg_count!("rand"; args == 2; range);

        let min = native_arg!("rand"; args @ 0 => Number @ range);
        let max = native_arg!("rand"; args @ 1 => Number @ range);

        let mut rng = rand::thread_rng();
        let result = rng.gen_range(*min..*max) as i64;

        Ok(Value::Number(result as f64).into())
    }

    fn std_fn_randf(args: Vec<Rc<Value>>, range: Range) -> IResult {
        native_arg_count!("randf"; args == 2; range);

        let min = native_arg!("randf"; args @ 0 => Number @ range);
        let max = native_arg!("randf"; args @ 1 => Number @ range);

        let mut rng = rand::thread_rng();
        let result = rng.gen_range(*min..*max);

        Ok(Value::Number(result).into())
    }

    fn std_fn_randl(args: Vec<Rc<Value>>, range: Range) -> IResult {
        native_arg_count!("randl"; args == 1; range);

        let list = native_arg!("randl"; args @ 0 => List @ range);

        let mut rng = rand::thread_rng();
        let result = rng.gen_range(0..list.len());

        Ok(list[result].clone())
    }

    // Stdlib: List
    fn std_fn_len(args: Vec<Rc<Value>>, range: Range) -> IResult {
        native_arg_count!("len"; args == 1; range);

        let value = args.get(0).unwrap();

        let result = match value.borrow() {
            Value::String(value) => value.len(),
            Value::List(value) => value.len(),
            Value::Record(value) => value.len(),
            _ => Err(InterpreterError::TypeMismatch(
                range,
                "string, list or record",
                typeof_value(value),
            ))?,
        };

        Ok(Value::Number(result as f64).into())
    }

    fn std_fn_join(args: Vec<Rc<Value>>, range: Range) -> IResult {
        native_arg_count!("join"; args == 2; range);

        let list = native_arg!("join"; args @ 0 => List @ range);
        let separator = native_arg!("join"; args @ 1 => String @ range);

        let mut result = EcoString::new();
        for (i, item) in list.iter().enumerate() {
            if i != 0 {
                result.push_str(separator);
            }

            match item.borrow() {
                Value::String(value) => result.push_str(value),
                _ => result.push_str(&item.to_string()),
            };
        }

        Ok(Value::String(result.into()).into())
    }

    fn std_fn_init(args: Vec<Rc<Value>>, range: Range) -> IResult {
        native_arg_count!("init"; args == 1; range);

        let mut list = native_arg!("init"; args @ 0 => List @ range).to_owned();

        if list.len() == 0 {
            return Err(InterpreterError::NativeFunctionError("init: list is empty"));
        }

        list = list[0..list.len() - 1].to_vec().into();

        Ok(Value::List(list).into())
    }

    fn std_fn_last(args: Vec<Rc<Value>>, range: Range) -> IResult {
        native_arg_count!("last"; args == 1; range);

        let list = native_arg!("last"; args @ 0 => List @ range);

        let last = match list.last() {
            Some(last) => last,
            None => return Err(InterpreterError::NativeFunctionError("last: list is empty")),
        };

        Ok(last.clone())
    }

    fn std_fn_head(args: Vec<Rc<Value>>, range: Range) -> IResult {
        native_arg_count!("head"; args == 1; range);

        let list = native_arg!("head"; args @ 0 => List @ range);

        let head = match list.first() {
            Some(head) => head,
            None => return Err(InterpreterError::NativeFunctionError("head: list is empty")),
        };

        Ok(head.clone())
    }

    fn std_fn_tail(args: Vec<Rc<Value>>, range: Range) -> IResult {
        native_arg_count!("tail"; args == 1; range);

        let mut list = native_arg!("tail"; args @ 0 => List @ range).to_owned();

        if list.len() == 0 {
            return Err(InterpreterError::NativeFunctionError("tail: list is empty"));
        }

        list = list[1..list.len()].to_vec().into();

        Ok(Value::List(list).into())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Closure {
    params: Vec<EcoString>,
    body: AstNode,
    env: Environment,
}

type Range = (usize, usize);

#[derive(Debug, Clone)]
pub enum InterpreterError {
    UndefinedVariable(EcoString, Range),
    NotCallable(AstNode, Range),
    NotIterable(Rc<Value>, Range),
    WrongArgumentCount(Range, usize, usize),
    TypeMismatch(Range, &'static str, &'static str),
    PatternTypeMismatch(Range, &'static str, &'static str),
    Redeclaration(EcoString, Option<Range>, Range),
    InvalidFunctionName(Range),
    InvalidFunctionParam(Range),
    NativeFunctionError(&'static str),
    NativeFunctionWrongArgumentCount(Range, usize, usize),
    NativeFunctionInvalidArgument(Range, usize, String, &'static str),
}

type IResult = Result<Rc<Value>, InterpreterError>;

pub struct Interpreter {
    any_value: Rc<Value>,
    nil_value: Rc<Value>,
    fmt_re: Regex,
    fmt_range: Option<Range>,
}

impl Interpreter {
    pub fn new() -> Self {
        let fmt_re = Regex::new(r#"\{([a-zA-Z_][a-zA-Z0-9_]*)\}"#).unwrap();

        Self {
            any_value: Rc::new(Value::Any),
            nil_value: Rc::new(Value::Nil),
            fmt_re,
            fmt_range: None,
        }
    }

    pub fn display_error(&mut self, files: SimpleFiles<&str, &String>, error: InterpreterError) {
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
                    _ => unreachable!("not callable: {:#?}", value),
                };

                Diagnostic::error()
                    .with_code("run")
                    .with_message(format!("Cannot call a {}.", typename))
                    .with_labels(vec![Label::primary(0, range.0..range.1)])
            }
            InterpreterError::NotIterable(value, range) => {
                let range = match self.fmt_range {
                    Some(fmt_range) => fmt_range,
                    None => range,
                };

                Diagnostic::error()
                    .with_code("run")
                    .with_message(format!("Cannot iterate over a {}.", typeof_value(&value)))
                    .with_labels(vec![Label::primary(0, range.0..range.1)])
            }
            InterpreterError::WrongArgumentCount(range, expected, got) => {
                let range = match self.fmt_range {
                    Some(fmt_range) => fmt_range,
                    None => range,
                };

                Diagnostic::error()
                    .with_code("run")
                    .with_message("Wrong number of arguments.")
                    .with_labels(vec![Label::primary(0, range.0..range.1)])
                    .with_notes(vec![format!(
                        "Expected {} {}, got {}.",
                        expected,
                        if expected == 1 {
                            "argument"
                        } else {
                            "arguments"
                        },
                        got
                    )])
            }
            InterpreterError::TypeMismatch(range, expected, got) => {
                let range = match self.fmt_range {
                    Some(fmt_range) => fmt_range,
                    None => range,
                };

                Diagnostic::error()
                    .with_code("run")
                    .with_message("Type mismatch.")
                    .with_labels(vec![Label::primary(0, range.0..range.1)])
                    .with_notes(vec![format!("Expected a {}, got a {}.", expected, got)])
            }
            InterpreterError::PatternTypeMismatch(range, expected, got) => {
                let range = match self.fmt_range {
                    Some(fmt_range) => fmt_range,
                    None => range,
                };

                Diagnostic::error()
                    .with_code("run")
                    .with_message("Pattern type mismatch.")
                    .with_labels(vec![Label::primary(0, range.0..range.1)])
                    .with_notes(vec![format!(
                        "Cannot test a {} against a {}.",
                        got, expected
                    )])
            }
            InterpreterError::Redeclaration(name, org_range, new_range) => {
                let new_range = match self.fmt_range {
                    Some(fmt_range) => fmt_range,
                    None => new_range,
                };

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
            InterpreterError::InvalidFunctionName(range) => {
                let range = match self.fmt_range {
                    Some(fmt_range) => fmt_range,
                    None => range,
                };

                Diagnostic::error()
                    .with_code("run")
                    .with_message("Invalid function name.")
                    .with_labels(vec![Label::primary(0, range.0..range.1)])
                    .with_notes(vec![format!("A function cannot be called `_`.")])
            }
            InterpreterError::InvalidFunctionParam(range) => {
                let range = match self.fmt_range {
                    Some(fmt_range) => fmt_range,
                    None => range,
                };

                Diagnostic::error()
                    .with_code("run")
                    .with_message("Invalid parameter name.")
                    .with_labels(vec![Label::primary(0, range.0..range.1)])
                    .with_notes(vec![format!("A parameter cannot be called `_`.")])
            }
            InterpreterError::NativeFunctionError(message) => Diagnostic::error()
                .with_code("run")
                .with_message(message)
                .with_labels(vec![]),
            InterpreterError::NativeFunctionWrongArgumentCount(range, expected, got) => {
                let range = match self.fmt_range {
                    Some(fmt_range) => fmt_range,
                    None => range,
                };

                Diagnostic::error()
                    .with_code("run")
                    .with_message("Wrong number of arguments.")
                    .with_labels(vec![Label::primary(0, range.0..range.1)])
                    .with_notes(vec![format!(
                        "Expected {} {}, got {}.",
                        expected,
                        if expected == 1 {
                            "argument"
                        } else {
                            "arguments"
                        },
                        got
                    )])
            }
            InterpreterError::NativeFunctionInvalidArgument(range, index, expected, got) => {
                let range = match self.fmt_range {
                    Some(fmt_range) => fmt_range,
                    None => range,
                };

                Diagnostic::error()
                    .with_code("run")
                    .with_message(format!("Invalid argument at index {}.", index))
                    .with_labels(vec![Label::primary(0, range.0..range.1)])
                    .with_notes(vec![format!("Expected a {}, got a {}.", expected, got)])
            }
        };

        term::emit(&mut writer.lock(), &config, &files, &diagnostic).unwrap();
    }

    pub fn eval(&mut self, node: &AstNode, environment: &mut Environment) -> IResult {
        let result = match node {
            AstNode::Prog { body, .. } | AstNode::Block { body, .. } => {
                let mut result = self.nil_value.clone();
                for node in body {
                    result = self.eval(node, environment)?;
                }
                result
            }
            AstNode::ExprStmt { expr, .. } => self.eval(expr, environment)?,
            AstNode::FuncDecl {
                id: id_node,
                is_const,
                params: fn_params,
                body,
                ..
            } => {
                let id = match id_node {
                    Some(id) => match id.borrow() {
                        AstNode::Ident { ref name, .. } => {
                            if *name == UNDERLINE {
                                return Err(InterpreterError::InvalidFunctionName(id.range()));
                            } else {
                                if let Some((_, range)) = environment.get(name) {
                                    return Err(InterpreterError::Redeclaration(
                                        name.clone(),
                                        Some(*range),
                                        id.range(),
                                    ));
                                }

                                Some(name)
                            }
                        }
                        _ => unreachable!(),
                    },
                    None => None,
                };

                let mut params = Vec::new();
                for param in fn_params {
                    match param {
                        AstNode::Ident { ref name, .. } => {
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
                    env: environment.clone(),
                };

                let value = if *is_const {
                    // TODO: Make it lazy
                    let value = self.apply_closure(id_node.as_deref(), &closure, vec![])?;
                    value.clone()
                } else {
                    Rc::new(Value::Closure(closure))
                };

                if let Some(id) = id {
                    environment.set(id, value, id_node.as_ref().unwrap().range());
                }

                self.nil_value.clone()
            }
            AstNode::Lambda {
                params: fn_params,
                body,
                ..
            } => {
                let mut params = Vec::new();
                for param in fn_params {
                    match param {
                        AstNode::Ident { ref name, .. } => {
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
                    env: environment.clone(),
                };

                Rc::new(Value::Closure(closure))
            }
            AstNode::Case { expr, cases, .. } => {
                // TODO: Require every case pattern to be unique
                // TODO: Require that, for a given expression, all possible cases are covered

                let case_expr = self.eval(expr, environment)?;

                for case in cases {
                    let (pattern, body) = match case {
                        AstNode::CaseBranch { pattern, body, .. } => (pattern, body),
                        _ => unreachable!(),
                    };

                    match pattern.as_ref() {
                        AstNode::CasePattern { kind, expr, .. } => match kind {
                            CasePatternKind::Expr => {
                                let expr = self.eval(&expr, environment)?;

                                if match expr.borrow() {
                                    Value::List(list) => {
                                        let expr_items = match case_expr.borrow() {
                                            Value::List(items) => items,
                                            _ => Err(InterpreterError::PatternTypeMismatch(
                                                pattern.range(),
                                                typeof_value(&case_expr),
                                                typeof_value(&expr),
                                            ))?,
                                        };

                                        let mut is_match = true;

                                        if list.len() != expr_items.len() {
                                            is_match = false;
                                        } else {
                                            for (item, expr_item) in
                                                list.iter().zip(expr_items.iter())
                                            {
                                                if **item == Value::Any {
                                                    continue;
                                                }

                                                if item != expr_item {
                                                    is_match = false;
                                                    break;
                                                }
                                            }
                                        }

                                        is_match
                                    }
                                    _ => case_expr == expr,
                                } {
                                    return self.eval(body, environment);
                                }
                            }
                            CasePatternKind::Or => {
                                let expr = self.eval(&expr, environment)?;
                                let list = match expr.borrow() {
                                    Value::List(list) => list,
                                    _ => unreachable!(),
                                };

                                for item in list.iter() {
                                    if **item == Value::Any || case_expr == *item {
                                        return self.eval(body, environment);
                                    }
                                }
                            }
                            CasePatternKind::Any => {
                                return self.eval(body, environment);
                            }
                        },
                        _ => unreachable!(),
                    }
                }

                self.nil_value.clone()
            }
            AstNode::FuncCall {
                callee,
                args: pre_args,
                ..
            } => {
                let mut args = Vec::new();
                for arg in pre_args {
                    let value = self.eval(arg, environment)?;
                    args.push(value);
                }

                let result = match callee.borrow() {
                    AstNode::Ident { name, range } => {
                        if let Some((value, _)) = environment.get(name) {
                            match value.borrow() {
                                Value::Closure(closure) => {
                                    // TODO: Should we handle recursion?
                                    let result =
                                        self.apply_closure(Some(callee.as_ref()), closure, args)?;
                                    return Ok(result);
                                }
                                Value::NativeFunction(method) => {
                                    let result = method(args, *range)?;
                                    return Ok(result);
                                }
                                _ => {
                                    if args.len() != 0 {
                                        return Err(InterpreterError::NotCallable(
                                            *callee.clone(),
                                            callee.range(),
                                        ));
                                    }
                                }
                            }

                            value.clone()
                        } else {
                            return Err(InterpreterError::UndefinedVariable(name.clone(), *range));
                        }
                    }
                    AstNode::FuncCall { .. } => {
                        let callee_closure = self.eval(callee.as_ref(), environment)?;
                        let callee_closure = match callee_closure.borrow() {
                            Value::Closure(closure) => closure,
                            _ => unreachable!(),
                        };

                        self.apply_closure(Some(callee.as_ref()), callee_closure, args)?
                    }
                    _ => {
                        return Err(InterpreterError::NotCallable(
                            *callee.clone(),
                            callee.range(),
                        ));
                    }
                };

                result
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

                let expr = self.eval(expr_node, environment)?;
                match expr.borrow() {
                    Value::List(items) => {
                        let mut result = Vec::new();

                        let env = environment.clone();
                        let closure = Closure {
                            params: vec![id.clone()],
                            body: *body.clone(),
                            env,
                        };

                        for item in items.iter() {
                            result.push(self.apply_closure(None, &closure, vec![item.clone()])?);
                        }

                        Rc::new(Value::List(result.into()))
                    }
                    Value::String(value) => {
                        let mut result = Vec::new();

                        let env = environment.clone();
                        let closure = Closure {
                            params: vec![id.clone()],
                            body: *body.clone(),
                            env,
                        };

                        for c in value.chars() {
                            result.push(self.apply_closure(
                                None,
                                &closure,
                                vec![Rc::new(Value::String(c.into()))],
                            )?);
                        }

                        Rc::new(Value::List(result.into()))
                    }
                    _ => {
                        return Err(InterpreterError::NotIterable(
                            expr.clone(),
                            expr_node.range(),
                        ))?;
                    }
                }
            }
            AstNode::Ident { name, range } => {
                if *name == UNDERLINE {
                    return Ok(self.any_value.clone());
                }

                if let Some((value, _)) = environment.get(name) {
                    match value.borrow() {
                        Value::Closure(closure) => {
                            let result = self.apply_closure(Some(node), closure, vec![])?;
                            return Ok(result);
                        }
                        Value::NativeFunction(method) => {
                            let result = method(vec![], *range)?;
                            return Ok(result);
                        }
                        _ => {}
                    }

                    return Ok(value.clone());
                } else {
                    return Err(InterpreterError::UndefinedVariable(
                        name.clone(),
                        node.range(),
                    ));
                }
            }
            AstNode::StrLit {
                value,
                is_format,
                range,
            } => {
                if *is_format {
                    self.eval_fmt_string(value, *range, environment)?
                } else {
                    Rc::new(Value::String(value.clone()))
                }
            }
            AstNode::NumLit { value, .. } => Rc::new(Value::Number(*value)),
            AstNode::List { items, .. } => {
                let mut result = Vec::new();

                for item in items {
                    let value = self.eval(item, environment)?;
                    result.push(value);
                }

                Rc::new(Value::List(result.into()))
            }
            AstNode::Record { keys, values, .. } => {
                let mut record = FxHashMap::default();

                for (key, value) in keys.iter().zip(values.iter()) {
                    let key = match key {
                        AstNode::StrLit { value, .. } => value.clone(),
                        _ => unreachable!(),
                    };
                    let value = self.eval(value, environment)?;

                    record.insert(key, value);
                }

                Rc::new(Value::Record(record))
            }
            AstNode::Range {
                mode,
                start: start_node,
                end: end_node,
                ..
            } => {
                let start = self.eval(start_node, environment)?;
                let start = match start.borrow() {
                    Value::Number(value) => value,
                    value => {
                        return Err(InterpreterError::TypeMismatch(
                            start_node.range(),
                            "number",
                            typeof_value(&value),
                        ))
                    }
                };

                let end = self.eval(end_node, environment)?;
                let end = match end.borrow() {
                    Value::Number(value) => value,
                    value => {
                        return Err(InterpreterError::TypeMismatch(
                            end_node.range(),
                            "number",
                            typeof_value(&value),
                        ))
                    }
                };

                // TODO: Make it lazy
                let mut result = Vec::new();

                match mode {
                    RangeMode::Inclusive => {
                        for i in (*start as usize)..=(*end as usize) {
                            result.push(Rc::new(Value::Number(i as f64)));
                        }
                    }
                    RangeMode::Exclusive => {
                        for i in (*start as usize)..(*end as usize) {
                            result.push(Rc::new(Value::Number(i as f64)));
                        }
                    }
                }

                Rc::new(Value::List(result.into()))
            }
            _ => todo!("eval {:?}", node),
        };

        Ok(result)
    }

    fn apply_closure(
        &mut self,
        callee: Option<&AstNode>,
        closure: &Closure,
        args: Vec<Rc<Value>>,
    ) -> IResult {
        if closure.params.len() != args.len() {
            let callee = match callee {
                Some(callee) => callee,
                None => unreachable!(),
            };

            return Err(InterpreterError::WrongArgumentCount(
                callee.range(),
                closure.params.len(),
                args.len(),
            ));
        }

        let mut env = closure.env.clone();
        for (param, arg) in closure.params.iter().zip(args.iter()) {
            env.set(param, arg.clone(), (0, 0));
        }

        self.eval(&closure.body, &mut env)
    }

    fn eval_fmt_string(
        &mut self,
        value: &EcoString,
        range: Range,
        environment: &mut Environment,
    ) -> IResult {
        let mut swaps = Vec::new();

        for capture in self.fmt_re.captures_iter(value) {
            let name = capture.get(1).unwrap().as_str().to_owned();

            let full_match = capture.get(0).unwrap();
            let range = (full_match.start(), full_match.end());

            swaps.push((name, range));
        }

        let mut result = EcoString::new();
        let mut chars = value.chars().peekable().enumerate();

        for (name, (start, end)) in swaps {
            while let Some((i, c)) = chars.next() {
                if i == start {
                    let name = AstNode::Ident {
                        name: EcoString::from(name.to_owned()),
                        range,
                    };

                    let value = self.eval(&name, environment)?;
                    let value = match value.borrow() {
                        Value::String(value) => value.clone(),
                        value => EcoString::from(value.to_string()),
                    };

                    result.push_str(&value);
                    continue;
                }

                if i == end - 1 {
                    break;
                } else if i > start {
                    continue;
                }

                result.push(c);
            }
        }

        while let Some((_, c)) = chars.next() {
            result.push(c);
        }

        Ok(Rc::new(Value::String(EcoString::from(result))))
    }
}
