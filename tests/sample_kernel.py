"""A simple, symphony-agnostic kernel fixture for the theme-generation tests.

A plain 8-bar C-major period declared in the item-list DSL. Nothing here references the symphony's
themes — it exists only to exercise the kernel loader / generator (loadable by dotted path as
``tests.sample_kernel``). Tests must not depend on scratch experiments; this is their own fixture.
"""
from generation.theme_gen import frame, harm, kernel, phrase, pin

KERNEL = kernel(
    frame(8, key="C", role="love", lower="A4", upper="D6", durations=(0.5, 0.75, 1.0, 1.5, 2.0)),
    pins=[
        pin(1, 1.0, ("E5", 1.5), ("B4", 0.5), ("C5", 2.0), label="sample head"),
        pin(8, 3.0, ("C5", 2.0), label="sample cadence"),
    ],
    harmony=[
        harm(1, "I", "C", "E", "G"),
        harm(4, "cadence region", "G", "B", "D"),
        harm(8, "I cadence", "C", "E", "G"),
    ],
    structure=[
        phrase(1, 4, "antecedent", cadence="half cadence"),
        phrase(5, 8, "consequent", cadence="authentic cadence"),
    ],
)
