"""Bulk verify student + parent endpoints against the running backend."""
import json
import urllib.request

BASE = "http://localhost:8000/api"

def post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path, data=data, method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=10).read())

def get(path, token):
    req = urllib.request.Request(
        BASE + path, method="GET",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    try:
        return json.loads(urllib.request.urlopen(req, timeout=10).read())
    except Exception as e:
        return {"_error": str(e)}

def listify(x):
    if isinstance(x, list): return x
    if isinstance(x, dict):
        for k in ("data", "items"):
            if isinstance(x.get(k), list): return x[k]
    return []

print("\n=== STUDENT ENDPOINTS ===")
students = ["karim@school.test", "salma@school.test",
            "student002@school.test", "student020@school.test",
            "student040@school.test", "student070@school.test",
            "student095@school.test"]

for email in students:
    r = post("/login", {"email": email, "password": "password123"})
    tok = r["token"]
    sid = r["user"]["role_id"]
    name = r["user"]["name"]
    att   = get(f"/student/{sid}/attendance", tok)
    sched = get(f"/student/{sid}/schedule",   tok)
    hwk   = get(f"/student/{sid}/homework",   tok)
    mrk   = get(f"/student/{sid}/marks",      tok)
    print(f"  {name:30s} <{email:30s}>")
    print(f"     attendance: {att.get('percent','?'):>3}%  ({att.get('present','?')}P/{att.get('absent','?')}A/{att.get('late','?')}L/{att.get('excused','?')}E)")
    print(f"     schedule:   {len(sched.get('data', [])):>3} slots")
    print(f"     homework:   {len(listify(hwk)):>3} items")
    print(f"     marks:      {len(listify(mrk)):>3} results")

print("\n=== PARENT ENDPOINTS ===")
parents = ["parent@school.test", "parent1@school.test",
           "parent5@school.test", "parent25@school.test",
           "parent68@school.test", "parent69@school.test"]

for email in parents:
    r = post("/login", {"email": email, "password": "password123"})
    tok = r["token"]
    pid = r["user"]["role_id"]
    name = r["user"]["name"]
    kids = get(f"/parent/{pid}/children", tok)
    kid_names = [k.get("name") for k in kids.get("children", [])]
    print(f"  {name:30s} <{email:30s}>  children: {kid_names}")
    for k in kids.get("children", [])[:1]:  # first child only — sanity check
        cid = k["student_id"]
        att   = get(f"/parent/{pid}/children/{cid}/attendance", tok)
        sched = get(f"/parent/{pid}/children/{cid}/schedule",   tok)
        bus   = get(f"/parent/{pid}/children/{cid}/bus/live-location", tok)
        print(f"     -> {k.get('name')}: attendance={att.get('percent','?')}%, schedule={len(sched.get('data', []))} slots, "
              f"bus={'ok' if bus.get('location') else 'missing'} mode={bus.get('mode','?')} activity={bus.get('activity','?')}")
