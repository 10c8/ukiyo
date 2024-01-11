mod interpreter;
mod lexer;
mod parser;
mod scanner;

use std::time::Instant;

use codespan_reporting::{
    files::SimpleFiles,
    term::{
        self,
        termcolor::{ColorChoice, StandardStream},
    },
};

use crate::{
    interpreter::{Environment, Interpreter},
    lexer::{Lexer, ToDiagnostic},
    parser::Parser,
};

fn main() {
    /*
        let example = r#"
    fruits -> ["apples", "bananas", "oranges", "limes"]
    flavors -> {
        "fruit"     = ["sweet", "sour", "bitter"],
        "vegetable" = ["bitter", "salty", "umami"],
        "meat"      = ["salty", "umami", "sweet"],
    }

    get_color fruit ->
        color -> case fruit of
            "apples"            => "red"
            "bananas" | "limes" => "yellow"
            "oranges"           => "orange"
            _                   => gen %% { "ctx" = ["give me the fruit color"], "temp" = 0.3 }

        $"{fruit} are {color}"

    fruit_colors ->
        case (len fruits) of
            1 => get_color (head fruits)
            _ =>
                start -> join ((init fruits) |f| get_color f) ", "
                end   -> get_color (last fruits)

                $"{start} and {end}"
    "#;
    */

    let example = r#"
fizzbuzz -> 0..<10 |x|
  case [(mod x 3), (mod x 5)] of
    [0, 0] => "FizzBuzz"
    [0, _] => "Fizz"
    [_, 0] => "Buzz"
    _      => x

join fizzbuzz "\n"
"#;

    let mut files = SimpleFiles::new();
    files.add("src", example);

    let writer = StandardStream::stderr(ColorChoice::Auto);
    let config = term::Config::default();

    let start = Instant::now();

    let mut lexer = Lexer::new(example);
    if let Err(err) = lexer.lex() {
        let diagnostic = err.to_diagnostic(&files);
        term::emit(&mut writer.lock(), &config, &files, &diagnostic).unwrap();
        return;
    }

    println!("Lexer took: {:?}", Instant::now() - start);

    // let mut lexer_copy = lexer.clone();

    let start = Instant::now();

    let mut parser = Parser::new(lexer);
    let result = parser.parse();
    if let Err(err) = result {
        // println!("--- Tokens:");
        // loop {
        //     let token = lexer_copy.next();
        //     print!("{:#?}\n", token);

        //     if let lexer::Token::EOF { .. } = token {
        //         break;
        //     }
        // }

        parser.display_error(files, err);
        return;
    }

    let ast = result.unwrap();

    println!("Parser took: {:?}", Instant::now() - start);

    // println!("\n--- AST:");
    // println!("{:#?}", ast);

    let start = Instant::now();

    let mut environment = Environment::new();
    environment.load_stdlib();

    let mut tail = Interpreter::new();
    let result = tail.eval(&ast, &mut environment);
    if let Err(err) = result {
        tail.display_error(files, err);
        return;
    }

    println!("Interpreter took: {:?}", Instant::now() - start);

    println!("\n{}", result.unwrap());
}
