# 04 — Wire Mixed-frequency State Feature Through Commodity Feature Pipeline

**What to build:** Make the commodity futures feature pipeline generate and consume the Mixed-frequency State Feature artifact using the same operational style as existing optional state feature artifacts.

**Blocked by:** 03 — Join Mixed-frequency State Feature Into Daily Feature Assembly.

**Status:** complete

- [x] The pipeline can materialize Mixed-frequency State Feature output for commodity futures.
- [x] The pipeline can include the generated mixed-frequency columns in downstream state candidate data.
- [x] Missing optional inputs and enabled-required inputs behave consistently with existing feature artifacts.
