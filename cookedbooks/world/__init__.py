"""The episode generator: a fictional company's clean books, distractors and the injection pipeline.

Entry point: `cookedbooks.world.generate.generate_episode`. Pipeline:
plan (injectors declare footprints) -> simulate (clean world, smaller by the
footprint) -> inject (recorded changes) -> order and assign IDs -> export.

Isolation: this package never imports `cookedbooks.verifier`, and the
verifier never imports it. Both follow DESIGN.md independently.
"""

GENERATOR_VERSION = "0.2.0"
