"""Avoid full image decoding when Swift only asks for PSD sequence lengths.

run_psd_topk.sh verifies every datum (including its image binding) before
Swift starts. Swift's AddLengthPreprocessor otherwise calls template.encode on
every row merely to obtain ``[len(input_ids)]``. For PSD that also rehashes and
re-encodes each image. Actual training samples still use the original template
encode and media checks in LazyLLMDataset.
"""

import logging


logger = logging.getLogger(__name__)


def fast_length_row(row, *, max_length):
    ids = row.get('input_ids')
    if not isinstance(ids, list) or not ids:
        raise ValueError('EviScene PSD datum is missing input_ids')
    if len(ids) > max_length:
        raise ValueError('PSD datum exceeds max_length; truncation is forbidden')
    topk = row.get('topk')
    if not isinstance(topk, int) or isinstance(topk, bool) or topk < 1:
        raise ValueError('EviScene PSD datum has invalid topk')
    # The strict verify-psd-datums gate has already checked every target and
    # media binding. This stage must not materialize images or sparse targets.
    row['lengths'] = [len(ids)]
    return row


def install_fast_psd_length(add_length_preprocessor=None):
    if add_length_preprocessor is None:
        from swift.dataset import AddLengthPreprocessor
        add_length_preprocessor = AddLengthPreprocessor
    cls = add_length_preprocessor
    if getattr(cls, '_eviscene_psd_fast_length_installed', False):
        return
    original = cls.preprocess

    def preprocess(self, row):
        if type(self.template).__name__ != 'PsdTopKTemplate':
            return original(self, row)
        return fast_length_row(row, max_length=getattr(self.template, 'max_length', None) or 131072)

    cls.preprocess = preprocess
    cls._eviscene_psd_fast_length_installed = True
    logger.warning('EviScene PSD fast length preprocessor installed; training encode remains unchanged')
