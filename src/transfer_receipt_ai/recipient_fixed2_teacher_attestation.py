"""Second-stage hard pins for one reviewed formal fixed2 teacher publication.

The first Windows pass deliberately leaves these values unset and may only
materialize/inspect a source-only candidate.  After its canonical contract
bytes, size, and semantic subject have been independently reviewed, a second
code change fills all three constants.  Public verification and canonical
overlay consumption fail closed until that second stage is present.

This module is intentionally separate from the producer implementation: the
candidate contract binds the producer code bytes, so writing reviewed pins
here must not invalidate the publication they attest.
"""

from __future__ import annotations


ATTESTED_FIXED2_CONTRACT_SHA256: str | None = None
ATTESTED_FIXED2_CONTRACT_SIZE_BYTES: int | None = None
ATTESTED_FIXED2_PRODUCER_SUBJECT_ID: str | None = None
