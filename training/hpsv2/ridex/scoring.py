from ridex.remote import canonical_remote


def component_at(info, index):
    return {"hpsv2": float(info["components"]["hpsv2"][index])}


def component_metrics(rows, prefix):
    canonical = sum(float(row["components"]["hpsv2"]) for row in rows) / len(rows)
    return {f"{prefix}_hpsv2": canonical * 100., f"{prefix}/hpsv2_canonical": canonical}


def canonical_scorer():
    remote = canonical_remote()

    def score(images, prompts, metadata):
        scores = remote(images, prompts)
        return scores, {"components": {"hpsv2": scores}}

    return score
