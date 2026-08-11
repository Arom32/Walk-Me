import type { Place } from "../types";

export default function PlacesRow({ places }: { places: Place[] }) {
  if (places.length === 0) return null;

  return (
    <div className="places-row scroll-thin">
      {places.map((p, i) => (
        <div key={i} className="place-card">
          <div className="place-card-head">
            <span className="place-card-name">{p.name ?? "이름 미상"}</span>
            {p.visit_type && <span className="place-card-badge">{p.visit_type}</span>}
          </div>
          {p.address && <p className="place-card-address">{p.address}</p>}
        </div>
      ))}
    </div>
  );
}
