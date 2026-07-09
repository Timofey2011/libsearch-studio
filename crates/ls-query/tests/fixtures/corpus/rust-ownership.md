Rust's ownership system gives every value exactly one owner; when the owner
goes out of scope the value is dropped. Assignments and function calls move
ownership by default, so use-after-move is a compile-time error instead of a
runtime crash.

Borrowing lets code use a value without taking ownership. The borrow checker
enforces the aliasing rule: any number of shared references, or exactly one
mutable reference, but never both at once. This statically eliminates data
races within safe code.

Lifetimes name how long references must remain valid. The compiler infers most
of them, but signatures sometimes need explicit lifetime parameters so the
borrow checker can prove a returned reference does not outlive the data it
points into — this is how dangling references are prevented at compile time.

Smart pointers round out the model: Box for heap ownership, Rc and Arc for
shared ownership by reference counting, and RefCell or Mutex to move the
aliasing check to runtime when the static rules are too strict.
