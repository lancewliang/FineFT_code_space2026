# Compatibility Notes: migrate-operator-futures-to-polars

No unresolved compatibility differences are known at spec time.

During implementation, if a pandas behavior appears incorrect or Polars cannot reproduce a schema/dtype detail exactly, record:

- file and function
- input condition
- previous pandas output
- proposed Polars output
- recommended decision
- user decision
