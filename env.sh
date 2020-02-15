VENV=.venv/elaenia
SYLPH_VGGISH_MODEL_CHECKPOINT_FILE=assets/vggish_model.ckpt
[ -d $VENV ] && source $VENV/bin/activate || {
    echo "ERROR: Expected virtualenv dir to exist: $VENV" 1>&2
}
[ -e $SYLPH_VGGISH_MODEL_CHECKPOINT_FILE ] && export SYLPH_VGGISH_MODEL_CHECKPOINT_FILE || {
    echo "ERROR: Expected VGGish checkpoint file to exist: $SYLPH_VGGISH_MODEL_CHECKPOINT_FILE" 1>&2
}
