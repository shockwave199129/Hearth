# NLP golden transcript regression
#
# Cases:  app/eval/nlp_golden_cases.json
# Runner: python3 -m app.eval.nlp_golden
# Update: python3 -m app.eval.nlp_golden --update
#
# Install path (setup):
#   Source: repo models/nlp  or  backend/bundled/nlp
#   Dest:   {MODELS_DIR}/nlp   (override with NLP_MODELS_DIR)
#   Hook:   app.setup.nlp_models.ensure_nlp_models  (via download_models / scripts/setup.py)
