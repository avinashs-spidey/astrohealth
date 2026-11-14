"""ASTROHEALTH backend: FastAPI + SQLAlchemy (SQLite default, PostgreSQL via DATABASE_URL) + scikit-learn.
All data is simulated. Not a medical device."""
import csv
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.orm import declarative_base, sessionmaker
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DB = os.getenv("DATABASE_URL") or ("sqlite:////tmp/astrohealth.db" if os.getenv("VERCEL") else "sqlite:///astrohealth.db")
if DB.startswith("postgres://"): DB = DB.replace("postgres://", "postgresql://", 1)
engine = create_engine(DB, connect_args={"check_same_thread": False} if DB.startswith("sqlite") else {})
Session = sessionmaker(bind=engine)
Base = declarative_base()
FEATS = ["hr", "sys", "dia", "spo2", "temp", "sleep", "stress", "fatigue", "mood",
         "bone_loss", "muscle_loss", "wbc", "crp", "rad"]
DOSE_BUDGET = 180.0  # simulated mission dose budget, mSv
MISSION_DAYS = 180


class Astronaut(Base):
    __tablename__ = "astronauts"
    id = Column(Integer, primary_key=True)
    name = Column(String); role = Column(String); age = Column(Integer); scenario = Column(Integer, default=0)


class Record(Base):
    __tablename__ = "records"
    id = Column(Integer, primary_key=True)
    astronaut_id = Column(Integer, index=True); day = Column(Integer)
    cum_rad = Column(Float); risk = Column(Float); risk_label = Column(String); anomaly = Column(Integer)
    for _f in FEATS:
        locals()[_f] = Column(Float)


class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True)
    astronaut_id = Column(Integer, index=True); day = Column(Integer); level = Column(String); message = Column(String)


class FlightNote(Base):
    __tablename__ = "flight_notes"
    id = Column(Integer, primary_key=True)
    astronaut_id = Column(Integer, index=True); day = Column(Integer); author = Column(String); note = Column(String)


class SpaceLocation(Base):
    __tablename__ = "space_locations"
    id = Column(Integer, primary_key=True)
    astronaut_id = Column(Integer, index=True); day = Column(Integer)
    zone = Column(String); module = Column(String)
    lat = Column(Float); lon = Column(Float); alt_km = Column(Float)
    suit_kpa = Column(Float); o2_pct = Column(Float); comms_delay_s = Column(Float)


class SosEvent(Base):
    __tablename__ = "sos_events"
    id = Column(Integer, primary_key=True)
    astronaut_id = Column(Integer, index=True); day = Column(Integer)
    severity = Column(String); reason = Column(String); location_tag = Column(String)
    lat = Column(Float); lon = Column(Float); alt_km = Column(Float)
    vitals_snapshot = Column(String); status = Column(String, default="ACTIVE")
    responded_by = Column(String, default=""); resolution_note = Column(String, default="")


Base.metadata.create_all(engine)
row = lambda o: {c.name: getattr(o, c.name) for c in o.__table__.columns}

# ---------------- ML ----------------
def score(f):
    f = dict(zip(FEATS, f)); p = 0
    p += 2 * (f["hr"] > 100 or f["hr"] < 50) + 2 * (f["sys"] > 140) + 3 * (f["spo2"] < 92) + 2 * (f["temp"] > 38)
    p += (f["sleep"] < 5.5) + (f["stress"] > 7) + (f["fatigue"] > 7) + (f["mood"] < 3.5)
    p += 2 * (f["bone_loss"] > 6) + (f["muscle_loss"] > 15) + (f["wbc"] > 11 or f["wbc"] < 3.5) + 2 * (f["crp"] > 5)
    return p


def synth(n, rng, k=1.0):
    g = lambda m, s: rng.normal(m, s * k, n)
    X = np.c_[g(68, 14), g(120, 14), g(76, 9), np.minimum(g(97, 2.2), 100), g(36.8, .5), g(6.8, 1.2),
              np.clip(g(4, 2), 0, 10), np.clip(g(4, 2), 0, 10), np.clip(g(6, 2), 0, 10),
              abs(g(2, 3)), abs(g(5, 5)), g(7, 2.5), abs(g(2, 2.5)), abs(g(.6, .3))]
    return X


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def load_dataset(filename, labeled=True):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    X, y, an = [], [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row_d in csv.DictReader(fh):
            X.append([float(row_d[f]) for f in FEATS])
            if labeled:
                y.append(int(row_d["risk_class"]))
                an.append(int(row_d["is_anomaly"]))
    return (np.array(X), np.array(y), np.array(an)) if labeled else np.array(X)


_rng = np.random.default_rng(7)
_tr = load_dataset("train_telemetry.csv", labeled=True)
if _tr is not None:
    _X, _y, _ = _tr
    TRAIN_SOURCE = "data/train_telemetry.csv"
else:
    _X = synth(8000, _rng)
    _y = np.array([0 if (s := score(r)) < 2 else 1 if s < 5 else 2 for r in _X])
    TRAIN_SOURCE = "synthetic_in_memory"
_base = load_dataset("baseline_nominal.csv", labeled=False)
RF = RandomForestClassifier(150, random_state=1, n_jobs=-1).fit(_X, _y)
ISO = IsolationForest(contamination=0.03, random_state=1).fit(_base if _base is not None else synth(3000, _rng, 0.6))
LABELS = ["low", "moderate", "high"]


def analyze(X):
    X = np.atleast_2d(X); P = np.zeros((len(X), 3))
    P[:, RF.classes_] = RF.predict_proba(X)
    risk = np.round(100 * (0.5 * P[:, 1] + P[:, 2]), 1)
    return risk, [LABELS[i] for i in P.argmax(1)], (ISO.predict(X) == -1).astype(int)


# ---------------- Simulation ----------------
def sim_day(a, d, rng):
    t = d / MISSION_DAYS
    v = dict(hr=66 + 3 * np.sin(d / 5) + rng.normal(0, 2.5), sys=118 + rng.normal(0, 4), dia=76 + rng.normal(0, 3),
             spo2=97.5 + rng.normal(0, .5), temp=36.7 + rng.normal(0, .15), sleep=7 - .8 * t + rng.normal(0, .5),
             stress=3 + 1.5 * t + rng.normal(0, .7), fatigue=3 + 1.5 * t + rng.normal(0, .7), mood=7 - 1.2 * t + rng.normal(0, .6),
             bone_loss=d / 30 * .95 + rng.normal(0, .15), muscle_loss=.04 * d + rng.normal(0, .3),
             wbc=6.5 + rng.normal(0, .5), crp=1.2 + abs(rng.normal(0, .5)), rad=.55 + abs(rng.normal(0, .08)))
    if 120 <= d <= 122: v["rad"] += 15  # solar particle event
    s = a.scenario
    if s == 1 and 95 <= d <= 120:  # cardiovascular event
        k = min(1, (d - 95) / 8); v["hr"] += 38 * k; v["sys"] += 32 * k; v["spo2"] -= 4.5 * k
    if s == 2 and 60 <= d <= 78:  # infection / immune shift
        v["wbc"] += 6.5; v["crp"] += 8; v["temp"] += 1.5; v["fatigue"] += 3
    if s == 3 and d >= 100:  # psychological decline
        k = min(1, (d - 100) / 30); v["stress"] += 4.5 * k; v["sleep"] -= 2.6 * k; v["mood"] -= 3.5 * k; v["fatigue"] += 3 * k
    if s == 4 and 80 <= d <= 115:  # multi-system EVA deconditioning & strain
        k = min(1, (d - 80) / 10); v["hr"] += 22 * k; v["bone_loss"] += 3.2 * k; v["muscle_loss"] += 7.5 * k; v["fatigue"] += 2.5 * k; v["crp"] += 3.5 * k
    for k_, hi in (("stress", 10), ("fatigue", 10), ("mood", 10)): v[k_] = float(np.clip(v[k_], 0, hi))
    v["spo2"] = min(v["spo2"], 100.0)
    return v


def make_alerts(s, aid, recs):
    prev, c80, c50 = "low", False, False
    for r in recs:
        if r.risk_label == "high" and prev != "high":
            s.add(Alert(astronaut_id=aid, day=r.day, level="critical", message=top_finding(r)))
        if r.rad >= 10:
            s.add(Alert(astronaut_id=aid, day=r.day, level="critical", message=f"Solar particle event: {r.rad:.1f} mSv in one day. Shelter in storm shelter."))
        if not c50 and r.cum_rad >= .5 * DOSE_BUDGET: c50 = True; s.add(Alert(astronaut_id=aid, day=r.day, level="warning", message="Cumulative dose reached 50% of mission budget."))
        if not c80 and r.cum_rad >= .8 * DOSE_BUDGET: c80 = True; s.add(Alert(astronaut_id=aid, day=r.day, level="critical", message="Cumulative dose reached 80% of mission budget."))
        prev = r.risk_label


def _save_series(s, aid, rows):
    s.query(Record).filter_by(astronaut_id=aid).delete(); s.query(Alert).filter_by(astronaut_id=aid).delete()
    X = np.array([[r[f] for f in FEATS] for r in rows]); risk, lab, an = analyze(X)
    recs, cum = [], 0
    for i, r in enumerate(rows):
        cum += r["rad"]
        recs.append(Record(astronaut_id=aid, day=i + 1, cum_rad=round(cum, 2), risk=float(risk[i]), risk_label=lab[i], anomaly=int(an[i]),
                           **{k: round(float(x), 2) for k, x in r.items()}))
    s.add_all(recs); make_alerts(s, aid, recs); s.commit()


def run_sim(s, a):
    rng = np.random.default_rng(100 + a.id)
    rows = [sim_day(a, d, rng) for d in range(1, MISSION_DAYS + 1)]
    _save_series(s, a.id, rows)


# ---------------- Offline medical assistant (rule-based protocols) ----------------
def findings(r):
    out = []
    if r.hr > 100 or r.sys > 140 or r.spo2 < 92:
        out.append(("Cardiovascular", f"HR {r.hr:.0f}, SBP {r.sys:.0f}, SpO2 {r.spo2:.1f}% outside nominal range.",
                    "Rest, repeat vitals in 15 min, check cabin O2/CO2, run ECG protocol, notify flight surgeon if persistent."))
    if r.wbc > 11 or r.crp > 5 or r.temp > 38:
        out.append(("Immune", f"WBC {r.wbc:.1f}, CRP {r.crp:.1f}, temp {r.temp:.1f}C suggest inflammation/infection.",
                    "Isolate if feasible, hydrate, review infection protocol, begin symptom log, consult flight surgeon."))
    if r.stress > 7 or r.sleep < 5.5 or r.mood < 3.5:
        out.append(("Mental wellness", f"Sleep {r.sleep:.1f}h, stress {r.stress:.1f}/10, mood {r.mood:.1f}/10.",
                    "Schedule rest block, adjust light cycle, hold crew check-in, request psychological support session."))
    if r.bone_loss > 6 or r.muscle_loss > 15:
        out.append(("Musculoskeletal", f"Bone loss {r.bone_loss:.1f}%, muscle loss {r.muscle_loss:.1f}% from baseline.",
                    "Increase resistive exercise, review calcium/vitamin D intake, log exercise compliance."))
    if r.cum_rad >= .8 * DOSE_BUDGET:
        out.append(("Radiation", f"Cumulative dose {r.cum_rad:.0f} mSv (budget {DOSE_BUDGET:.0f}).", "Limit EVA, use shielded areas, review dose plan with mission control."))
    return out


top_finding = lambda r: (findings(r) or [("", "Elevated composite health risk.", "")])[0][1]


class Manual(BaseModel):
    day: int | None = None
    hr: float = 68; sys: float = 120; dia: float = 76; spo2: float = 97; temp: float = 36.8; sleep: float = 7
    stress: float = 4; fatigue: float = 4; mood: float = 6; bone_loss: float = 2; muscle_loss: float = 5
    wbc: float = 7; crp: float = 1.5; rad: float = .6


class NewAstronaut(BaseModel):
    name: str; role: str = "Mission Specialist"; age: int = 38; scenario: int = 0


app = FastAPI(title="ASTROHEALTH")


@app.on_event("startup")
def seed():
    with Session() as s:
        if s.query(Astronaut).count() == 0:
            for n, r, ag, sc in [("Cdr. Priya Nair", "Commander", 44, 0), ("Dr. Arjun Rao", "Flight Engineer", 39, 1),
                                 ("Meera Iyer", "Medical Officer", 36, 2), ("Karthik Subbu", "Pilot", 41, 3)]:
                a = Astronaut(name=n, role=r, age=ag, scenario=sc); s.add(a); s.flush(); run_sim(s, a)


def get_a(s, aid):
    a = s.get(Astronaut, aid)
    if not a: raise HTTPException(404, "Astronaut not found")
    return a


def rec_at(s, aid, day):
    q = s.query(Record).filter_by(astronaut_id=aid)
    if day: q = q.filter(Record.day <= day)
    return q.order_by(Record.day.desc()).first()


@app.get("/api/astronauts")
def astronauts():
    with Session() as s: return [row(a) for a in s.query(Astronaut).all()]


@app.post("/api/astronauts")
def add_astronaut(n: NewAstronaut):
    with Session() as s:
        a = Astronaut(**n.model_dump()); s.add(a); s.flush(); run_sim(s, a); return row(a)


ZONES = {
    "LEO â€” ISS Orbital Habitat": ("Destiny Lab", 418.0, 101.3, 21.0, 0.3),
    "EVA â€” External Truss / Airlock": ("Quest Airlock / P6 Truss", 418.0, 29.6, 100.0, 0.4),
    "Lunar Orbit â€” Gateway": ("HALO Module", 3400.0, 101.3, 21.0, 1.3),
    "Lunar Surface â€” Shackleton Rim": ("Hab Alpha / South Pole", 0.0, 56.5, 34.0, 1.3),
    "Deep Space â€” Mars Transit": ("Habitat Ring / Storm Shelter", 0.0, 101.3, 21.0, 180.0),
}


def resolve_location(s, a, day):
    loc = s.query(SpaceLocation).filter(SpaceLocation.astronaut_id == a.id, SpaceLocation.day <= day).order_by(SpaceLocation.day.desc(), SpaceLocation.id.desc()).first()
    if loc:
        return {**row(loc), "tagged": True}
    zone_names = list(ZONES.keys())
    zone = zone_names[(a.id - 1) % len(zone_names)]
    if zone == "EVA â€” External Truss / Airlock" and day % 5 != 0:
        zone = "LEO â€” ISS Orbital Habitat"
    mod, alt, kpa, o2, delay = ZONES[zone]
    if "Lunar Surface" in zone:
        lat, lon = -89.5, round(129.2 + 0.1 * np.sin(day), 2)
    elif "Deep Space" in zone:
        lat, lon = round(12.4 * np.cos(day / 25), 2), round(((day * 4.5) % 360) - 180, 2)
        delay = round(12.0 + 2.2 * day, 1)
    else:
        lat = round(51.6 * np.sin(day * 0.7 + a.id * 1.1), 2)
        lon = round(((day * 37 + a.id * 73) % 360) - 180, 2)
    return {"astronaut_id": a.id, "day": day, "zone": zone, "module": mod, "lat": float(lat), "lon": float(lon),
            "alt_km": alt, "suit_kpa": kpa, "o2_pct": o2, "comms_delay_s": delay, "tagged": False}


@app.delete("/api/astronauts/{aid}")
def delete_astronaut(aid: int):
    with Session() as s:
        a = get_a(s, aid)
        if s.query(Astronaut).count() <= 1: raise HTTPException(400, "At least one crew member must remain")
        for cls in (Record, Alert, FlightNote, SpaceLocation, SosEvent): s.query(cls).filter_by(astronaut_id=aid).delete()
        s.delete(a); s.commit(); return {"ok": True}


@app.post("/api/simulate/{aid}")
def simulate(aid: int):
    with Session() as s: run_sim(s, get_a(s, aid)); return {"ok": True}


@app.get("/api/astronauts/{aid}/records")
def records(aid: int):
    with Session() as s: return [row(r) for r in s.query(Record).filter_by(astronaut_id=aid).order_by(Record.day)]


@app.post("/api/astronauts/{aid}/records")
def add_record(aid: int, m: Manual):
    with Session() as s:
        get_a(s, aid); last = rec_at(s, aid, None); day = m.day or ((last.day + 1) if last else 1)
        prev = rec_at(s, aid, day - 1); d = m.model_dump(exclude={"day"})
        s.query(Record).filter_by(astronaut_id=aid, day=day).delete()
        risk, lab, an = analyze([d[f] for f in FEATS])
        r = Record(astronaut_id=aid, day=day, cum_rad=round((prev.cum_rad if prev else 0) + m.rad, 2), risk=float(risk[0]), risk_label=lab[0], anomaly=int(an[0]), **d)
        s.add(r); make_alerts_single(s, aid, r); s.commit(); return row(r)


def make_alerts_single(s, aid, r):
    if r.risk_label == "high": s.add(Alert(astronaut_id=aid, day=r.day, level="critical", message=top_finding(r)))


@app.get("/api/mission")
def mission(day: int = MISSION_DAYS):
    with Session() as s:
        out = []
        for a in s.query(Astronaut).all():
            recs = s.query(Record).filter(Record.astronaut_id == a.id, Record.day <= day).order_by(Record.day).all()
            r = recs[-1] if recs else None
            bflag = bool(len(recs) >= 15 and personal_series(recs)[1][-1] > THR)
            n = s.query(Alert).filter(Alert.astronaut_id == a.id, Alert.day <= day).count()
            rd = compute_readiness(r, bflag) if r else None
            loc = resolve_location(s, a, day)
            sos = s.query(SosEvent).filter(SosEvent.astronaut_id == a.id, SosEvent.status == "ACTIVE").order_by(SosEvent.id.desc()).first()
            out.append({**row(a), "latest": row(r) if r else None, "alerts": n, "baseline_flag": bflag,
                        "readiness": {"status": rd["status"], "score": rd["score"]} if rd else None,
                        "location": loc, "active_sos": row(sos) if sos else None})
        return out


@app.get("/api/alerts")
def alerts(astronaut_id: int | None = None, day: int = MISSION_DAYS):
    with Session() as s:
        q = s.query(Alert).filter(Alert.day <= day)
        if astronaut_id: q = q.filter_by(astronaut_id=astronaut_id)
        return [row(a) for a in q.order_by(Alert.day.desc()).limit(50)]


@app.get("/api/astronauts/{aid}/assistant")
def assistant(aid: int, day: int = MISSION_DAYS):
    with Session() as s:
        a = get_a(s, aid); r = rec_at(s, aid, day)
        if not r: raise HTTPException(404, "No data")
        wk = s.query(Record).filter(Record.astronaut_id == aid, Record.day <= r.day, Record.day > r.day - 7).all()
        base = s.query(Record).filter(Record.astronaut_id == aid, Record.day <= 14).all()
        trend = []
        for f, lbl in (("hr", "Heart rate"), ("sleep", "Sleep"), ("stress", "Stress")):
            if wk and base:
                d = np.mean([getattr(x, f) for x in wk]) - np.mean([getattr(x, f) for x in base])
                if abs(d) > .12 * max(abs(np.mean([getattr(x, f) for x in base])), 1): trend.append(f"{lbl} 7-day average is {d:+.1f} vs. early-mission baseline.")
        f = findings(r)
        return {"astronaut": a.name, "day": r.day, "risk": r.risk, "risk_label": r.risk_label, "anomaly": bool(r.anomaly),
                "findings": [{"system": x, "observation": y, "action": z} for x, y, z in f], "trends": trend,
                "summary": ("No protocol thresholds exceeded. Continue routine monitoring." if not f else f"{len(f)} area(s) need attention; follow listed protocols."),
                "disclaimer": "Simulated decision support based on fixed protocol thresholds. Not a medical diagnosis."}


@app.get("/api/astronauts/{aid}/report", response_class=PlainTextResponse)
def report(aid: int, day: int = MISSION_DAYS):
    with Session() as s:
        a = get_a(s, aid); r = rec_at(s, aid, day); al = s.query(Alert).filter(Alert.astronaut_id == aid, Alert.day <= day).order_by(Alert.day).all()
        if not r: raise HTTPException(404, "No data")
        loc = resolve_location(s, a, r.day)
        sos = s.query(SosEvent).filter(SosEvent.astronaut_id == aid, SosEvent.status == "ACTIVE").order_by(SosEvent.id.desc()).first()
        L = [f"ASTROHEALTH MEDICAL REPORT", f"Astronaut: {a.name} ({a.role}, age {a.age})", f"Mission day: {r.day}/{MISSION_DAYS}",
             f"Space location: {loc['zone']} | {loc['module']} (Lat {loc['lat']}, Lon {loc['lon']}, Alt {loc['alt_km']} km, Comms delay {loc['comms_delay_s']}s)",
             f"SOS Beacon: {'ACTIVE - ' + sos.reason if sos else 'none'}", "",
             f"Risk score: {r.risk}/100 ({r.risk_label}); anomaly flag: {'YES' if r.anomaly else 'no'}", "", "Latest indicators:"]
        L += [f"  {f:12s} {getattr(r, f)}" for f in FEATS] + [f"  cumulative dose {r.cum_rad} mSv of {DOSE_BUDGET:.0f} budget", "", "Findings:"]
        L += [f"  [{x}] {y}\n      Action: {z}" for x, y, z in findings(r)] or ["  none"]
        L += ["", f"Alert history ({len(al)}):"] + [f"  Day {x.day} [{x.level}] {x.message}" for x in al]
        L += ["", "Simulated data. Not a medical device."]
        return "\n".join(L)


# ---------------- v2: personalized baselines, drift detection, forecasting, benchmark ----------------
FAST = ["hr", "sys", "dia", "spo2", "temp", "sleep", "stress", "fatigue", "mood", "wbc", "crp"]
THR = 2.2  # rms z-score threshold on 3-day smoothed deviation from the astronaut's OWN baseline
EVENTS = {1: (95, 120), 2: (60, 78), 3: (100, 180), 4: (80, 115)}  # ground-truth windows of injected conditions


def personal_series(recs):
    A = np.array([[getattr(r, f) for f in FAST] for r in recs]); base = A[:14]
    mu = base.mean(0); sd = np.maximum(base.std(0), .005 * np.abs(mu) + .05)
    Z = (A - mu) / sd; inst = np.sqrt((Z ** 2).mean(1))
    sm = np.convolve(np.r_[[inst[0]] * 2, inst], np.ones(3) / 3, "valid")
    return Z, sm


def forecast(recs, remaining):
    out = []
    for f, thr, w, lower in (("bone_loss", 6, 21, False), ("muscle_loss", 15, 21, False), ("cum_rad", DOSE_BUDGET, 21, False),
                             ("sleep", 5.5, 14, True), ("stress", 7, 14, False)):
        y = np.array([getattr(r, f) for r in recs[-w:]])
        if len(y) < 5: continue
        m, b = np.polyfit(np.arange(len(y)), y, 1); cur = float(m * (len(y) - 1) + b)
        dist, slope = (cur - thr, -m) if lower else (thr - cur, m)
        d = 0.0 if dist <= 0 else (dist / slope if slope > 1e-6 else None)
        out.append(dict(feature=f, current=round(cur, 2), slope=round(float(m), 3), threshold=thr,
                        days_to_threshold=None if d is None else round(float(d), 1),
                        breach_before_mission_end=bool(d is not None and d <= remaining)))
    return out


@app.get("/api/astronauts/{aid}/insights")
def insights(aid: int, day: int = MISSION_DAYS):
    with Session() as s:
        get_a(s, aid)
        recs = s.query(Record).filter(Record.astronaut_id == aid, Record.day <= day).order_by(Record.day).all()
        if len(recs) < 15: return {"ready": False}
        Z, sm = personal_series(recs); z = Z[-1]
        drivers = [{"feature": FAST[i], "z": round(float(z[i]), 1), "value": getattr(recs[-1], FAST[i])} for i in np.argsort(-np.abs(z))[:3]]
        return {"ready": True, "threshold": THR, "flag": bool(sm[-1] > THR), "drivers": drivers,
                "series": [{"day": r.day, "score": round(float(v), 2)} for r, v in zip(recs, sm)],
                "forecast": forecast(recs, MISSION_DAYS - recs[-1].day)}


@app.get("/api/benchmark")
def benchmark():
    """Compare detectors against the injected ground-truth event windows (synthetic data)."""
    with Session() as s:
        tot = {k: [0, 0, 0] for k in ("population", "personalized", "fused")}; rows = []
        for a in s.query(Astronaut).all():
            recs = s.query(Record).filter_by(astronaut_id=a.id).order_by(Record.day).all()
            if len(recs) < 15: continue
            _, sm = personal_series(recs); ev = EVENTS.get(a.scenario)
            truth = np.array([bool(ev) and ev[0] <= r.day <= ev[1] for r in recs])
            pop = np.array([r.risk_label == "high" for r in recs]); per = sm > THR
            for k, fl in (("population", pop), ("personalized", per), ("fused", pop | per)):
                tp, fp, fn = int((fl & truth).sum()), int((fl & ~truth).sum()), int((~fl & truth).sum())
                for i, v in enumerate((tp, fp, fn)): tot[k][i] += v
                hit = [r.day for r, f in zip(recs, fl) if f and ev and r.day >= ev[0]]
                rows.append({"astronaut": a.name, "model": k, "event_start": ev[0] if ev else None,
                             "detection_delay_days": (hit[0] - ev[0]) if hit else None, "false_alarm_days": fp})
        res = {}
        for k, (tp, fp, fn) in tot.items():
            p = tp / (tp + fp) if tp + fp else 0; r_ = tp / (tp + fn) if tp + fn else 0
            res[k] = {"precision": round(p, 3), "recall": round(r_, 3), "f1": round(2 * p * r_ / (p + r_), 3) if p + r_ else 0}
        return {"summary": res, "per_astronaut": rows,
                "note": "Synthetic data with injected events; validate on real datasets before clinical claims."}


# ---------------- v3: EVA readiness, countermeasure simulator, flight surgeon Q&A, notes, CSV ----------------
def compute_readiness(r, baseline_flag=False):
    checks = []
    def add(dom, st, detail):
        checks.append({"domain": dom, "status": st, "detail": detail})

    if r.hr > 100 or r.hr < 50 or r.sys > 140 or r.spo2 < 92:
        add("Cardiovascular", "NO-GO", f"HR {r.hr:.0f} bpm, SBP {r.sys:.0f} mmHg, SpO2 {r.spo2:.1f}% out of EVA limits.")
    elif r.hr > 90 or r.sys > 132 or r.spo2 < 94:
        add("Cardiovascular", "CAUTION", f"Borderline vitals (HR {r.hr:.0f}, SBP {r.sys:.0f}, SpO2 {r.spo2:.1f}%).")
    else:
        add("Cardiovascular", "GO", f"Nominal (HR {r.hr:.0f}, SBP {r.sys:.0f}, SpO2 {r.spo2:.1f}%).")

    if r.temp > 38 or r.wbc > 11 or r.wbc < 3.5 or r.crp > 5:
        add("Immune / Thermal", "NO-GO", f"Temp {r.temp:.1f}C, WBC {r.wbc:.1f}, CRP {r.crp:.1f} indicate active inflammation.")
    elif r.temp > 37.5 or r.crp > 3:
        add("Immune / Thermal", "CAUTION", f"Mild elevation (Temp {r.temp:.1f}C, CRP {r.crp:.1f}).")
    else:
        add("Immune / Thermal", "GO", f"Clear (Temp {r.temp:.1f}C, WBC {r.wbc:.1f}, CRP {r.crp:.1f}).")

    if r.sleep < 5 or r.stress > 8 or r.fatigue > 8 or r.mood < 3:
        add("Neurocognitive", "NO-GO", f"Sleep {r.sleep:.1f}h, stress {r.stress:.1f}, fatigue {r.fatigue:.1f} impair EVA safety.")
    elif r.sleep < 6 or r.stress > 6.5 or r.fatigue > 6.5 or r.mood < 4:
        add("Neurocognitive", "CAUTION", f"Moderate strain (sleep {r.sleep:.1f}h, stress {r.stress:.1f}, fatigue {r.fatigue:.1f}).")
    else:
        add("Neurocognitive", "GO", f"Rested (sleep {r.sleep:.1f}h, stress {r.stress:.1f}, mood {r.mood:.1f}).")

    margin = round(max(0.0, DOSE_BUDGET - r.cum_rad), 1)
    if r.rad >= 5 or r.cum_rad >= .9 * DOSE_BUDGET:
        add("Radiation", "NO-GO", f"Daily {r.rad:.1f} mSv / cumulative {r.cum_rad:.1f} mSv ({margin} mSv budget left).")
    elif r.cum_rad >= .7 * DOSE_BUDGET:
        add("Radiation", "CAUTION", f"Cumulative {r.cum_rad:.1f} mSv ({margin} mSv budget remaining).")
    else:
        add("Radiation", "GO", f"Dose {r.cum_rad:.1f}/{DOSE_BUDGET:.0f} mSv ({margin} mSv margin).")

    if r.risk_label == "high":
        add("ML & Baseline", "NO-GO", f"High composite risk ({r.risk}/100).")
    elif r.risk_label == "moderate" or r.anomaly or baseline_flag:
        add("ML & Baseline", "CAUTION", f"Risk {r.risk}/100, anomaly={'yes' if r.anomaly else 'no'}, drift={'yes' if baseline_flag else 'no'}.")
    else:
        add("ML & Baseline", "GO", f"Low risk ({r.risk}/100), on personal baseline.")

    sts = [c["status"] for c in checks]
    overall = "NO-GO" if "NO-GO" in sts else ("CAUTION" if "CAUTION" in sts else "GO")
    score_val = max(0, 100 - 25 * sts.count("NO-GO") - 10 * sts.count("CAUTION"))
    return {"status": overall, "score": score_val, "eva_dose_margin_msv": margin, "checks": checks}


@app.get("/api/astronauts/{aid}/readiness")
def readiness(aid: int, day: int = MISSION_DAYS):
    with Session() as s:
        a = get_a(s, aid)
        recs = s.query(Record).filter(Record.astronaut_id == aid, Record.day <= day).order_by(Record.day).all()
        if not recs: raise HTTPException(404, "No data")
        bflag = bool(len(recs) >= 15 and personal_series(recs)[1][-1] > THR)
        return {"astronaut": a.name, "day": recs[-1].day, **compute_readiness(recs[-1], bflag)}


class Intervention(BaseModel):
    kind: str  # exercise_boost | sleep_reset | storm_shelter | medical_stabilize | inject_solar_event
    day: int = 1
    duration: int = 14


@app.post("/api/astronauts/{aid}/intervene")
def intervene(aid: int, body: Intervention):
    valid_kinds = ("exercise_boost", "sleep_reset", "storm_shelter", "medical_stabilize", "inject_solar_event")
    if body.kind not in valid_kinds:
        raise HTTPException(400, "Unknown intervention kind")
    with Session() as s:
        get_a(s, aid)
        recs = s.query(Record).filter_by(astronaut_id=aid).order_by(Record.day).all()
        if not recs: raise HTTPException(404, "No data")
        rows = [{f: getattr(r, f) for f in FEATS} for r in recs]
        start = max(1, min(body.day, len(rows))) - 1
        end = min(len(rows), start + max(1, min(body.duration, 60)))
        for idx in range(start, end):
            r = rows[idx]
            if body.kind == "exercise_boost":
                r["bone_loss"] = max(0.0, r["bone_loss"] - 1.6); r["muscle_loss"] = max(0.0, r["muscle_loss"] - 3.8); r["mood"] = min(10.0, r["mood"] + 0.4)
            elif body.kind == "sleep_reset":
                r["sleep"] = min(9.0, r["sleep"] + 1.4); r["stress"] = max(0.0, r["stress"] - 1.8); r["fatigue"] = max(0.0, r["fatigue"] - 1.5); r["mood"] = min(10.0, r["mood"] + 1.2)
            elif body.kind == "storm_shelter":
                r["rad"] = round(r["rad"] * 0.2, 2)
            elif body.kind == "medical_stabilize":
                r["hr"] = 0.5 * r["hr"] + 0.5 * 68; r["sys"] = 0.5 * r["sys"] + 0.5 * 118; r["spo2"] = min(100.0, 0.5 * r["spo2"] + 0.5 * 98)
                r["temp"] = 0.5 * r["temp"] + 0.5 * 36.7; r["crp"] = 0.5 * r["crp"] + 0.5 * 1.2; r["wbc"] = 0.5 * r["wbc"] + 0.5 * 6.5
            elif body.kind == "inject_solar_event" and idx < start + 3:
                r["rad"] += 15.0
        _save_series(s, aid, rows)
        return {"ok": True, "applied": body.kind, "from_day": start + 1, "to_day": end}


class AskQuery(BaseModel):
    question: str
    day: int = MISSION_DAYS


@app.post("/api/astronauts/{aid}/ask")
def ask_assistant(aid: int, body: AskQuery):
    with Session() as s:
        a = get_a(s, aid)
        recs = s.query(Record).filter(Record.astronaut_id == aid, Record.day <= body.day).order_by(Record.day).all()
        if not recs: raise HTTPException(404, "No data")
        r = recs[-1]; q = body.question.lower()
        bflag, drivers = False, []
        if len(recs) >= 15:
            Z, sm = personal_series(recs); z = Z[-1]; bflag = bool(sm[-1] > THR)
            drivers = [f"{FAST[i]} ({z[i]:+.1f}Ïƒ)" for i in np.argsort(-np.abs(z))[:3]]
        rd = compute_readiness(r, bflag); f = findings(r); fc = forecast(recs, MISSION_DAYS - r.day)
        if any(w in q for w in ("eva", "spacewalk", "ready", "clear", "go")):
            blockers = [f"{c['domain']}: {c['detail']}" for c in rd["checks"] if c["status"] != "GO"]
            ans = f"{a.name} is {rd['status']} for EVA on day {r.day} (readiness {rd['score']}/100). " + (
                "Blockers/cautions: " + "; ".join(blockers) if blockers else "All 5 readiness domains are nominal.")
        elif any(w in q for w in ("rad", "solar", "dose", "storm")):
            rad_fc = next((x for x in fc if x["feature"] == "cum_rad"), None)
            eta = f"Projected to hit {DOSE_BUDGET:.0f} mSv in {rad_fc['days_to_threshold']}d." if rad_fc and rad_fc["days_to_threshold"] is not None else "Cumulative dose slope is currently stable."
            ans = f"Day {r.day} dose: {r.rad:.2f} mSv/day, cumulative {r.cum_rad:.1f}/{DOSE_BUDGET:.0f} mSv ({rd['eva_dose_margin_msv']} mSv left). {eta}"
        elif any(w in q for w in ("forecast", "trend", "limit", "breach", "future")):
            risks = [f"{x['feature']} ({x['days_to_threshold']}d)" for x in fc if x["breach_before_mission_end"]]
            ans = ("Projected breaches before day 180: " + ", ".join(risks) + ".") if risks else "No tracked metrics are projected to breach limits before mission end."
        elif any(w in q for w in ("action", "countermeasure", "treat", "do", "recommend")):
            acts = [z for _, _, z in f]
            ans = ("Recommended protocols: " + " | ".join(acts)) if acts else "No protocol thresholds exceeded; maintain routine exercise, hydration, and sleep schedule."
        elif any(w in q for w in ("loc", "where", "zone", "orbit", "sos", "comm", "delay")):
            loc = resolve_location(s, a, r.day)
            sos = s.query(SosEvent).filter(SosEvent.astronaut_id == a.id, SosEvent.status == "ACTIVE").order_by(SosEvent.id.desc()).first()
            ans = (f"{a.name} is tagged at {loc['zone']} ({loc['module']}), coordinates ({loc['lat']}, {loc['lon']}), alt {loc['alt_km']} km, "
                   f"suit/cabin {loc['suit_kpa']} kPa / {loc['o2_pct']}% O2, one-way comms delay {loc['comms_delay_s']}s. "
                   + (f"ACTIVE SOS: {sos.reason}." if sos else "No active SOS distress beacon."))
        else:
            drv = (", top baseline drivers: " + ", ".join(drivers)) if drivers else ""
            fnd = "; ".join(y for _, y, _ in f) if f else "No protocol thresholds exceeded."
            ans = f"Day {r.day} composite risk is {r.risk}/100 ({r.risk_label}), EVA status {rd['status']} ({rd['score']}/100){drv}. {fnd}"
        return {"answer": ans, "day": r.day, "readiness": rd["status"]}


class NewNote(BaseModel):
    day: int = MISSION_DAYS
    author: str = "Flight Surgeon"
    note: str


@app.get("/api/astronauts/{aid}/notes")
def get_notes(aid: int):
    with Session() as s:
        get_a(s, aid)
        return [row(n) for n in s.query(FlightNote).filter_by(astronaut_id=aid).order_by(FlightNote.day.desc(), FlightNote.id.desc()).all()]


@app.post("/api/astronauts/{aid}/notes")
def add_note(aid: int, body: NewNote):
    with Session() as s:
        get_a(s, aid)
        if not body.note.strip(): raise HTTPException(400, "Note cannot be empty")
        n = FlightNote(astronaut_id=aid, day=body.day, author=body.author.strip() or "Flight Surgeon", note=body.note.strip())
        s.add(n); s.commit(); return row(n)


@app.get("/api/astronauts/{aid}/export.csv")
def export_csv(aid: int, day: int = MISSION_DAYS):
    with Session() as s:
        a = get_a(s, aid)
        recs = s.query(Record).filter(Record.astronaut_id == aid, Record.day <= day).order_by(Record.day).all()
        cols = ["day", *FEATS, "cum_rad", "risk", "risk_label", "anomaly"]
        lines = [",".join(cols)] + [",".join(str(getattr(r, c)) for c in cols) for r in recs]
        fname = f"astrohealth_{a.name.lower().replace(' ', '_').replace('.', '')}_day{day}.csv"
        return PlainTextResponse("\n".join(lines), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{fname}"'})


# ---------------- v4: space location tagging, SOS distress beacon, detailed vitals & triage ----------------
class LocationTag(BaseModel):
    day: int = MISSION_DAYS
    zone: str
    module: str = ""
    lat: float | None = None
    lon: float | None = None
    alt_km: float | None = None
    suit_kpa: float | None = None
    o2_pct: float | None = None
    comms_delay_s: float | None = None


@app.get("/api/astronauts/{aid}/location")
def get_location(aid: int, day: int = MISSION_DAYS):
    with Session() as s:
        a = get_a(s, aid)
        return {**resolve_location(s, a, day), "zones": list(ZONES.keys())}


@app.post("/api/astronauts/{aid}/location")
def tag_location(aid: int, body: LocationTag):
    with Session() as s:
        a = get_a(s, aid)
        preset = ZONES.get(body.zone, ("Custom Module", 400.0, 101.3, 21.0, 1.0))
        cur = resolve_location(s, a, body.day)
        loc = SpaceLocation(
            astronaut_id=aid, day=body.day, zone=body.zone, module=body.module.strip() or preset[0],
            lat=body.lat if body.lat is not None else cur["lat"],
            lon=body.lon if body.lon is not None else cur["lon"],
            alt_km=body.alt_km if body.alt_km is not None else preset[1],
            suit_kpa=body.suit_kpa if body.suit_kpa is not None else preset[2],
            o2_pct=body.o2_pct if body.o2_pct is not None else preset[3],
            comms_delay_s=body.comms_delay_s if body.comms_delay_s is not None else preset[4],
        )
        s.add(loc)
        s.add(FlightNote(astronaut_id=aid, day=body.day, author="Navigation", note=f"Location tagged: {loc.zone} â€” {loc.module} ({loc.lat}, {loc.lon}, alt {loc.alt_km} km)."))
        s.commit()
        return {**row(loc), "tagged": True, "zones": list(ZONES.keys())}


@app.get("/api/mission/locations")
def mission_locations(day: int = MISSION_DAYS):
    with Session() as s:
        crew = []
        for a in s.query(Astronaut).all():
            loc = resolve_location(s, a, day)
            sos = s.query(SosEvent).filter(SosEvent.astronaut_id == a.id, SosEvent.status == "ACTIVE").order_by(SosEvent.id.desc()).first()
            crew.append({"id": a.id, "name": a.name, "role": a.role, **loc, "active_sos": row(sos) if sos else None})
        return {"day": day, "zones": {k: {"module": v[0], "alt_km": v[1], "suit_kpa": v[2], "o2_pct": v[3], "comms_delay_s": v[4]} for k, v in ZONES.items()}, "crew": crew}


class SosTrigger(BaseModel):
    day: int = MISSION_DAYS
    severity: str = "MAYDAY"
    reason: str = "Medical / EVA distress beacon activated"


@app.post("/api/astronauts/{aid}/sos")
def trigger_sos(aid: int, body: SosTrigger):
    with Session() as s:
        a = get_a(s, aid); r = rec_at(s, aid, body.day)
        loc = resolve_location(s, a, body.day)
        snap = f"HR {r.hr:.0f}, SpO2 {r.spo2:.1f}%, SBP {r.sys:.0f}, Temp {r.temp:.1f}C, Risk {r.risk}" if r else "No telemetry"
        tag = f"{loc['zone']} Â· {loc['module']}"
        ev = SosEvent(astronaut_id=aid, day=body.day, severity=body.severity.upper(), reason=body.reason.strip() or "Distress beacon activated",
                      location_tag=tag, lat=loc["lat"], lon=loc["lon"], alt_km=loc["alt_km"], vitals_snapshot=snap, status="ACTIVE")
        s.add(ev)
        s.add(Alert(astronaut_id=aid, day=body.day, level="critical", message=f"SOS {ev.severity} at {tag}: {ev.reason}"))
        s.commit()
        return {**row(ev), "astronaut_name": a.name}


@app.get("/api/sos")
def list_sos(status: str | None = None, astronaut_id: int | None = None):
    with Session() as s:
        names = {a.id: a.name for a in s.query(Astronaut).all()}
        q = s.query(SosEvent)
        if status: q = q.filter(SosEvent.status == status.upper())
        if astronaut_id: q = q.filter(SosEvent.astronaut_id == astronaut_id)
        return [{**row(e), "astronaut_name": names.get(e.astronaut_id, f"Crew #{e.astronaut_id}")} for e in q.order_by(SosEvent.id.desc()).limit(30)]


class SosRespond(BaseModel):
    status: str = "ACKNOWLEDGED"  # ACKNOWLEDGED | RESOLVED
    responded_by: str = "Mission Control"
    resolution_note: str = ""


@app.post("/api/sos/{sos_id}/respond")
def respond_sos(sos_id: int, body: SosRespond):
    with Session() as s:
        ev = s.get(SosEvent, sos_id)
        if not ev: raise HTTPException(404, "SOS event not found")
        st = body.status.upper()
        if st not in ("ACKNOWLEDGED", "RESOLVED"): raise HTTPException(400, "Invalid SOS status")
        ev.status = st; ev.responded_by = body.responded_by.strip() or "Mission Control"; ev.resolution_note = body.resolution_note.strip()
        s.add(FlightNote(astronaut_id=ev.astronaut_id, day=ev.day, author=ev.responded_by,
                         note=f"SOS #{ev.id} {st}: {ev.resolution_note or ev.reason}"))
        s.commit()
        return row(ev)


REFS = {
    "hr": ("bpm", 50, 100, "Heart rate"), "sys": ("mmHg", 90, 140, "Systolic BP"), "dia": ("mmHg", 60, 90, "Diastolic BP"),
    "spo2": ("%", 92, 100, "Oxygen sat"), "temp": ("Â°C", 36.0, 38.0, "Core temp"), "sleep": ("h", 5.5, 9.0, "Sleep"),
    "stress": ("/10", 0, 7.0, "Stress"), "fatigue": ("/10", 0, 7.0, "Fatigue"), "mood": ("/10", 3.5, 10.0, "Mood"),
    "bone_loss": ("%", 0, 6.0, "Bone loss"), "muscle_loss": ("%", 0, 15.0, "Muscle loss"),
    "wbc": ("K/ÂµL", 3.5, 11.0, "WBC"), "crp": ("mg/L", 0, 5.0, "CRP"),
    "rad": ("mSv/d", 0, 5.0, "Daily rad"), "cum_rad": ("mSv", 0, DOSE_BUDGET, "Cumul. dose"),
}


@app.get("/api/astronauts/{aid}/vitals_detail")
def vitals_detail(aid: int, day: int = MISSION_DAYS):
    with Session() as s:
        a = get_a(s, aid)
        recs = s.query(Record).filter(Record.astronaut_id == aid, Record.day <= day).order_by(Record.day).all()
        if not recs: raise HTTPException(404, "No data")
        r = recs[-1]; base = recs[:min(14, len(recs))]; loc = resolve_location(s, a, r.day)
        sos = s.query(SosEvent).filter(SosEvent.astronaut_id == aid, SosEvent.status == "ACTIVE").first()
        items = []
        for k, (unit, lo, hi, lbl) in REFS.items():
            val = float(getattr(r, k)); bval = float(np.mean([getattr(x, k) for x in base]))
            st = "critical" if (val < lo or val > hi) else ("borderline" if (val < lo + .08 * (hi - lo) or val > hi - .08 * (hi - lo)) else "nominal")
            items.append({"key": k, "label": lbl, "value": round(val, 2), "unit": unit, "range": f"{lo}â€“{hi}",
                          "delta_vs_base": round(val - bval, 2), "status": st})
        radar = {
            "labels": ["Cardiovascular", "Respiratory", "Immune", "Neuro / Sleep", "Musculoskeletal", "Radiation"],
            "values": [
                round(float(min(100, max(0, abs(r.hr - 68) * 1.8 + max(0, r.sys - 120) * 1.5))), 1),
                round(float(min(100, max(0, (98 - r.spo2) * 12))), 1),
                round(float(min(100, max(0, max(0, r.wbc - 8) * 10 + r.crp * 8 + max(0, r.temp - 37) * 35))), 1),
                round(float(min(100, max(0, max(0, 7.5 - r.sleep) * 20 + r.stress * 6 + r.fatigue * 5))), 1),
                round(float(min(100, max(0, r.bone_loss * 10 + r.muscle_loss * 4.5))), 1),
                round(float(min(100, max(0, (r.cum_rad / DOSE_BUDGET) * 100 + r.rad * 3))), 1),
            ],
        }
        prio = "P1 â€” IMMEDIATE" if (sos or r.risk_label == "high") else ("P2 â€” URGENT" if (r.risk_label == "moderate" or r.anomaly) else "P3 â€” ROUTINE")
        comms = "Autonomous onboard crew protocol required (high light-time comms delay)." if loc["comms_delay_s"] >= 10 else "Real-time Mission Control / Flight Surgeon voice loop available."
        return {"day": r.day, "triage": prio, "comms_mode": comms, "location": loc, "items": items, "radar": radar}


@app.get("/api/model/eval")
def model_eval():
    te = load_dataset("test_telemetry.csv", labeled=True)
    if te is None:
        raise HTTPException(404, "data/test_telemetry.csv not found")
    X_te, y_te, an_te = te
    y_pred = RF.predict(X_te)
    iso_pred = (ISO.predict(X_te) == -1).astype(int)
    rep = classification_report(y_te, y_pred, target_names=LABELS, output_dict=True, zero_division=0)
    return {
        "train_source": TRAIN_SOURCE, "train_rows": int(len(_X)), "test_source": "data/test_telemetry.csv", "test_rows": int(len(X_te)),
        "accuracy": round(float(accuracy_score(y_te, y_pred)), 4),
        "macro_f1": round(float(rep["macro avg"]["f1-score"]), 4),
        "per_class": {k: {"precision": round(rep[k]["precision"], 3), "recall": round(rep[k]["recall"], 3), "f1": round(rep[k]["f1-score"], 3), "support": int(rep[k]["support"])} for k in LABELS},
        "confusion_matrix": confusion_matrix(y_te, y_pred, labels=[0, 1, 2]).tolist(),
        "anomaly_flagged_rate": round(float(iso_pred.mean()), 4),
        "extreme_anomaly_recall": round(float(iso_pred[an_te == 1].mean()), 4) if (an_te == 1).any() else None,
    }


if not os.getenv("VERCEL"):
    app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "public"), html=True), name="static")
