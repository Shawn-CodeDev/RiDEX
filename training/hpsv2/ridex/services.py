from ridex.reward_protocol import PROTOCOL


def check_services(document, session):
    if document.get("profile") != "hpsv2" or set(document.get("urls", {})) != {"hpsv2"}:
        raise ValueError("Expected a single-HPSv2 readiness document")
    urls = document["urls"]["hpsv2"]
    if not isinstance(urls, list) or not urls or any(not isinstance(u, str) or not u.strip() for u in urls):
        raise ValueError("Empty or invalid HPSv2 URL pool")
    for url in urls:
        response = session.get(url.rstrip("/")+"/health", timeout=(5, 10))
        response.raise_for_status()
        health = response.json()
        if health.get("reward_type") != "hpsv2" or health.get("reward_protocol") != PROTOCOL:
            raise ValueError("Incorrect canonical HPSv2 service contract")
    return {"HPSV2_REWARD_URLS": ",".join(urls)}
