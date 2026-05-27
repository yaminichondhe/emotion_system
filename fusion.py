def fuse_emotions(face, voice):
    f_em, f_conf = face
    v_em, v_conf = voice

    if f_conf > v_conf:
        return {
            "emotion": f_em,
            "face": f_em,
            "voice": v_em,
            "confidence": float(f_conf)
        }

    return {
        "emotion": v_em,
        "face": f_em,
        "voice": v_em,
        "confidence": float(v_conf)
    }