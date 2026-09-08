"""Defense Method Evaluation (Section 4.2, Table 4).

Evaluates nine defense methods from the paper against SafeGrad ladders.

Implemented defenses (Table 4):
  negative_prompt   : Prepend safety-focused negative prompt to generation
  guidance_scale_low: Reduce CFG guidance scale to 3.0 (reduces concept
                      adherence for both safe and unsafe prompts)
  llm_rewrite       : Rewrite prompt via safety-focused LLM transformation
                      (heuristic approximation — see note below)

Note on defense method implementations
----------------------------------------
Table 4 of the paper includes nine defense methods:
  MACE, TRCE, SafetyDPO, SLD, NegativePrompt, NSFW-Detox,
  SuppressEO, SafeGen, and UCE.

MACE, TRCE, SafetyDPO, UCE, SafeGen, and SuppressEO require separate
fine-tuned checkpoints or modified model architectures that are not
redistributable as open-source.  These methods are approximated here
as negative-prompt injection to allow benchmark reproducibility with
publicly available models.  If you have access to the original checkpoints,
replace the corresponding method implementations in eval/benchmark/defense_eval.py.

SLD and NSFW-Detox are partially approximated via guidance-scale reduction.

Paper reference: Section 4.2, Table 4
"""

from eval.benchmark.defense_eval import compute  # noqa: F401

__all__ = ["compute"]
