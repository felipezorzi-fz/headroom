"""``_get_cache_prices`` must not bill cache slices it cannot authoritatively price.

LiteLLM omits ``cache_read_input_token_cost`` / ``cache_creation_input_token_cost``
for the long tail of its priced models (most Bedrock, Mistral, Fireworks and
OpenAI-compatible gateway models). The old ``.get(field, uncached)`` default
billed those cache reads at the full uncached rate, so ``totals()`` (and the
/stats figures it feeds) over-charged every cache-warm request on those models.

The fix fails closed: a missing cache field is priced at ``$0`` to reconcile
with the canonical ``litellm.cost_per_token`` path (which books $0 for a slice
it cannot price), rather than fabricating a provider-wide discount or premium.
Bedrock in particular fronts many vendors whose cache economics are not
interchangeable, so no provider heuristic is applied. Explicit LiteLLM cache
fields still win.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from headroom.proxy.cost import CostTracker


def _patch_litellm(monkeypatch: pytest.MonkeyPatch, model_cost: dict) -> None:
    monkeypatch.setattr(
        "headroom.proxy.cost._get_litellm_module",
        lambda: SimpleNamespace(model_cost=model_cost),
    )
    monkeypatch.setattr(
        "headroom.pricing.litellm_pricing.resolve_litellm_model",
        lambda model: model,
    )


class TestGetCachePrices:
    def test_uses_explicit_litellm_cache_fields_when_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_litellm(
            monkeypatch,
            {
                "m": {
                    "input_cost_per_token": 1e-6,
                    "cache_read_input_token_cost": 1e-7,
                    "cache_creation_input_token_cost": 1.25e-6,
                    "litellm_provider": "anthropic",
                }
            },
        )
        assert CostTracker()._get_cache_prices("m") == (1e-7, 1.25e-6, 1e-6)

    def test_missing_cache_fields_priced_at_zero_not_full_rate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A model with an input price but no cache fields: exactly the 1600+
        # long-tail models the bug affects. Before the fix both slices billed at
        # the full uncached rate; now they fail closed at $0, reconciling with
        # what litellm.cost_per_token books for the same slice.
        _patch_litellm(
            monkeypatch,
            {"m": {"input_cost_per_token": 5e-7, "litellm_provider": "mistral"}},
        )
        assert CostTracker()._get_cache_prices("m") == (0.0, 0.0, 5e-7)

    def test_only_the_missing_field_is_zeroed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Explicit read cost present, write cost absent: keep the real read,
        # fail closed on the write.
        _patch_litellm(
            monkeypatch,
            {
                "m": {
                    "input_cost_per_token": 1e-6,
                    "cache_read_input_token_cost": 3e-7,
                    "litellm_provider": "openai",
                }
            },
        )
        assert CostTracker()._get_cache_prices("m") == (3e-7, 0.0, 1e-6)

    def test_no_input_price_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_litellm(monkeypatch, {"m": {"litellm_provider": "anthropic"}})
        assert CostTracker()._get_cache_prices("m") is None

    @pytest.mark.parametrize(
        "litellm_provider",
        ["bedrock", "bedrock_converse", "mistral", "fireworks_ai", "cohere_chat", None, "made-up"],
    )
    def test_heterogeneous_and_unknown_providers_do_not_inherit_anthropic_rates(
        self, monkeypatch: pytest.MonkeyPatch, litellm_provider
    ) -> None:
        # Regression: a missing cache field must never be back-filled with
        # Anthropic's 0.1x/1.25x economics. Bedrock fronts Anthropic, Amazon,
        # Meta, Mistral, Cohere and others; unknown OpenAI-compatible providers
        # are likewise not Anthropic-priced.
        uncached = 4e-6
        info = {"input_cost_per_token": uncached}
        if litellm_provider is not None:
            info["litellm_provider"] = litellm_provider
        _patch_litellm(monkeypatch, {"m": info})
        cache_read, cache_write, got_uncached = CostTracker()._get_cache_prices("m")
        assert got_uncached == uncached
        assert cache_read == 0.0
        assert cache_write == 0.0
        # Explicitly not the Anthropic multipliers the old fallback would apply.
        assert cache_read != pytest.approx(uncached * 0.1)
        assert cache_write != pytest.approx(uncached * 1.25)

    def test_missing_cache_slices_match_canonical_zero_pricing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The missing cache slices are priced at $0, the same amount the canonical
        litellm.cost_per_token path books for a slice LiteLLM cannot price, so the
        cache breakdown reconciles with the primary accounting instead of diverging.
        """
        _patch_litellm(
            monkeypatch,
            {"m": {"input_cost_per_token": 5e-7, "litellm_provider": "fireworks_ai"}},
        )
        cr_price, cw_price, uncached_price = CostTracker()._get_cache_prices("m")
        cr_tokens, cw_tokens = 20_000, 5_000
        # Cache-slice cost contributed to totals() is exactly $0, matching canonical.
        assert cr_tokens * cr_price + cw_tokens * cw_price == 0.0
        assert uncached_price == 5e-7
