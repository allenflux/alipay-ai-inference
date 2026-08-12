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


ATTESTED_FIXED2_CONTRACT_SHA256: str | None = (
    "07b225570ed763bb7f73dc8c053fb30a2d1dae0f4f20134c25871dcf5d215c74"
)
ATTESTED_FIXED2_CONTRACT_SIZE_BYTES: int | None = 62561
ATTESTED_FIXED2_PRODUCER_SUBJECT_ID: str | None = (
    "c592e97e24d9b043fdc60f2acbc38be995d0e86f152e5e30611214ba94fae9e7"
)
