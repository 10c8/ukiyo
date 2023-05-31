#[derive(Clone)]
pub struct Scanner {
    cursor: usize,
    column: usize,
    line: usize,
    chars: Vec<char>,
}

impl Scanner {
    pub fn new(input: &str) -> Scanner {
        Scanner {
            cursor: 0,
            column: 1,
            line: 1,
            chars: input.chars().collect(),
        }
    }

    /// Returns the current column position.
    pub fn column(&self) -> usize {
        self.column
    }

    /// Returns the current line position.
    pub fn line(&self) -> usize {
        self.line
    }

    /// Returns the lookahead character without advancing the cursor.
    pub fn peek(&self) -> Option<&char> {
        self.chars.get(self.cursor)
    }

    /// Returns the nth lookahead character without advancing the cursor.
    pub fn peek_nth(&self, n: usize) -> Option<&char> {
        self.chars.get(self.cursor + n)
    }

    /// If there is a next character, returns it and advances the cursor.
    pub fn next(&mut self) -> Option<&char> {
        match self.chars.get(self.cursor) {
            Some(chr) => {
                self.cursor += 1;

                if chr == &'\n' {
                    self.line += 1;
                    self.column = 1;
                } else {
                    self.column += 1;
                }

                Some(chr)
            }
            None => None,
        }
    }

    /// Returns true if `target` is found at the current cursor position and advances the cursor.
    /// Otherwise, returns false and does not advance the cursor.
    pub fn try_consume(&mut self, target: char) -> bool {
        match self.chars.get(self.cursor) {
            Some(chr) if *chr == target => {
                self.cursor += 1;

                if chr == &'\n' {
                    self.line += 1;
                    self.column = 1;
                } else {
                    self.column += 1;
                }

                true
            }
            _ => false,
        }
    }

    /// Returns true if `target` is found at the current cursor position and advances the cursor.
    /// Otherwise, returns false and does not advance the cursor.
    pub fn try_consume_sequence(&mut self, target: &str) -> bool {
        let mut cursor = self.cursor;
        let mut line = self.line;
        let mut column = self.column;

        for chr in target.chars() {
            match self.chars.get(cursor) {
                Some(c) if *c == chr => {
                    cursor += 1;

                    if c == &'\n' {
                        line += 1;
                        column = 1;
                    } else {
                        column += 1;
                    }
                }
                _ => return false,
            }
        }

        self.cursor = cursor;
        self.line = line;
        self.column = column;

        true
    }

    /// Advances the cursor while `predicate` returns true.
    pub fn consume_while<F>(&mut self, mut predicate: F)
    where
        F: FnMut(&char) -> bool,
    {
        while let Some(chr) = self.chars.get(self.cursor) {
            if predicate(chr) {
                self.next();
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_scanner() {
        let input = "abc\ndef";
        let mut scanner = super::Scanner::new(input);

        assert_eq!(scanner.column(), 1);
        assert_eq!(scanner.line(), 1);

        assert_eq!(scanner.peek(), Some(&'a'));
        assert_eq!(scanner.next(), Some(&'a'));
        assert_eq!(scanner.column(), 2);
        assert_eq!(scanner.line(), 1);

        assert_eq!(scanner.next(), Some(&'b'));
        assert_eq!(scanner.column(), 3);
        assert_eq!(scanner.line(), 1);

        assert_eq!(scanner.next(), Some(&'c'));
        assert_eq!(scanner.column(), 4);
        assert_eq!(scanner.line(), 1);

        assert_eq!(scanner.next(), Some(&'\n'));
        assert_eq!(scanner.column(), 1);
        assert_eq!(scanner.line(), 2);

        assert_eq!(scanner.peek(), Some(&'d'));
        assert_eq!(scanner.next(), Some(&'d'));
        assert_eq!(scanner.column(), 2);
        assert_eq!(scanner.line(), 2);

        assert_eq!(scanner.next(), Some(&'e'));
        assert_eq!(scanner.column(), 3);
        assert_eq!(scanner.line(), 2);

        assert_eq!(scanner.next(), Some(&'f'));
        assert_eq!(scanner.column(), 4);
        assert_eq!(scanner.line(), 2);

        assert_eq!(scanner.next(), None);
        assert_eq!(scanner.column(), 4);
        assert_eq!(scanner.line(), 2);
    }

    #[test]
    fn test_try_consume() {
        let input = "abc\ndef\nghi";
        let mut scanner = super::Scanner::new(input);

        assert_eq!(scanner.try_consume('a'), true);
        assert_eq!(scanner.try_consume('b'), true);
        assert_eq!(scanner.try_consume('c'), true);
        assert_eq!(scanner.try_consume('d'), false);
        assert_eq!(scanner.try_consume('e'), false);
        assert_eq!(scanner.try_consume('f'), false);
        assert_eq!(scanner.try_consume('\n'), true);
        assert_eq!(scanner.try_consume('d'), true);
        assert_eq!(scanner.try_consume('e'), true);
        assert_eq!(scanner.try_consume('f'), true);
        assert_eq!(scanner.try_consume('\n'), true);
        assert_eq!(scanner.try_consume('g'), true);
        assert_eq!(scanner.try_consume('h'), true);
        assert_eq!(scanner.try_consume('i'), true);
    }

    #[test]
    fn test_try_consume_sequence() {
        let input = "abc\ndef\nghi";
        let mut scanner = super::Scanner::new(input);

        assert_eq!(scanner.try_consume_sequence("abc"), true);
        assert_eq!(scanner.try_consume_sequence("def"), false);
        assert_eq!(scanner.try_consume_sequence("\ndef\n"), true);
        assert_eq!(scanner.try_consume_sequence("ghi"), true);
    }

    #[test]
    fn test_consume_while() {
        let input = "abc\ndef\nghi";
        let mut scanner = super::Scanner::new(input);

        scanner.consume_while(|chr| chr.is_alphabetic());
        assert_eq!(scanner.column(), 4);
        assert_eq!(scanner.line(), 1);

        scanner.consume_while(|chr| *chr == '\n');
        assert_eq!(scanner.column(), 1);
        assert_eq!(scanner.line(), 2);

        scanner.consume_while(|chr| chr.is_alphabetic());
        assert_eq!(scanner.column(), 4);
        assert_eq!(scanner.line(), 2);

        scanner.consume_while(|chr| *chr == '\n');
        assert_eq!(scanner.column(), 1);
        assert_eq!(scanner.line(), 3);

        scanner.consume_while(|chr| chr.is_alphabetic());
        assert_eq!(scanner.column(), 4);
        assert_eq!(scanner.line(), 3);
    }
}
