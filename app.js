// data/projects.json을 읽어 대표 프로젝트 카드, 그 외 목록, 상세 모달을 그린다.
// projects.json은 portfolio-hub의 scripts/export_site_data.py가 만든다. 직접 고치지 말 것.

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const child of children.flat()) if (child) node.append(child);
  return node;
}

const IMG = (file) => `assets/img/${file}`;

function metaText(p) {
  const parts = [];
  if (p.period) parts.push(p.period);
  parts.push(p.ownership === "team" ? `팀 ${p.team_size ?? ""}명`.replace(" 명", "") : "개인");
  return parts.join(" · ");
}

function card(p) {
  const img = p.images?.[0];
  return el("button", { class: "card", type: "button", "aria-haspopup": "dialog", onclick: () => openModal(p) }, [
    img && el("div", { class: "thumb" }, [el("img", { src: IMG(img.file), alt: `${p.name} ${img.caption}`, loading: "lazy" })]),
    el("div", { class: "card-body" }, [
      p.award && el("span", { class: "award", text: p.award }),
      el("h3", { text: p.name }),
      p.subtitle && el("p", { class: "sub", text: p.subtitle }),
      el("p", { class: "meta", text: `${metaText(p)} · ${p.role}` }),
      el("p", { class: "desc", text: p.one_liner }),
      p.stack && el("ul", { class: "tags" }, p.stack.slice(0, 6).map((s) => el("li", { text: s }))),
      el("span", { class: "more", text: "자세히 보기 →" }),
    ]),
  ]);
}

function row(p) {
  return el("li", {}, [
    el("button", { class: "row", type: "button", "aria-haspopup": "dialog", onclick: () => openModal(p) }, [
      el("span", { class: "row-title", text: `${p.name} — ${p.subtitle ?? ""}` }),
      el("span", { class: "row-period", text: p.period }),
      el("span", { class: "row-sub", text: p.role }),
    ]),
  ]);
}

function section(title, children) {
  return el("section", { class: "m-section" }, [el("h4", { text: title }), ...children]);
}

function modalContent(p) {
  const links = [];
  for (const r of p.repos ?? []) {
    links.push(r.url
      ? el("a", { class: "btn primary", href: r.url, target: "_blank", rel: "noopener", text: `GitHub · ${r.label}` })
      : el("span", { class: "private", text: `비공개 레포 · ${r.label}` }));
  }
  if (p.notion) links.push(el("a", { class: "btn", href: p.notion, target: "_blank", rel: "noopener", text: "상세 (Notion)" }));
  if (p.demo) links.push(el("a", { class: "btn", href: p.demo, target: "_blank", rel: "noopener", text: "Demo" }));

  return [
    p.award && el("span", { class: "award", text: p.award }),
    el("h3", { class: "m-title", id: "modal-title", text: p.name }),
    p.subtitle && el("p", { class: "m-sub", text: p.subtitle }),
    el("div", { class: "m-meta" }, [el("span", { text: metaText(p) }), el("span", { text: p.role })]),
    el("p", { text: p.one_liner }),
    p.images?.length && section("구조", p.images.map((img) =>
      el("figure", {}, [
        el("a", { href: IMG(img.file), target: "_blank", rel: "noopener" }, [
          el("img", { src: IMG(img.file), alt: `${p.name} ${img.caption}`, loading: "lazy" }),
        ]),
        el("figcaption", { text: img.caption }),
      ]))),
    p.highlights?.length && section("핵심 구현", [el("ul", {}, p.highlights.map((h) => el("li", { text: h })))]),
    p.troubleshooting?.length && section("문제 해결", [el("div", { class: "trouble" }, p.troubleshooting.map((t) =>
      el("div", {}, [el("b", { text: t.problem }), el("span", { text: t.solution })])))]),
    p.stack?.length && section("기술 스택", [el("ul", { class: "tags" }, p.stack.map((s) => el("li", { text: s })))]),
    links.length && section("링크", [el("div", { class: "m-links" }, links),
      p.ownership === "team" && el("p", { class: "note", text: "팀 프로젝트입니다. 위 내용은 본인이 담당한 부분입니다." })]),
  ];
}

const dialog = document.getElementById("modal");
const body = document.getElementById("modal-body");
dialog.querySelector(".close").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); });
dialog.addEventListener("close", () => history.replaceState(null, "", location.pathname));

function openModal(p) {
  body.replaceChildren(...modalContent(p).filter(Boolean));
  if (!dialog.open) dialog.showModal();
  dialog.scrollTop = 0;
  history.replaceState(null, "", `#${p.id}`);
}

async function main() {
  try {
    const res = await fetch("data/projects.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(res.status);
    const { projects } = await res.json();
    document.getElementById("featured").replaceChildren(...projects.filter((p) => p.featured).map(card));
    document.getElementById("others").replaceChildren(...projects.filter((p) => !p.featured).map(row));
    const hit = projects.find((p) => `#${p.id}` === location.hash);
    if (hit) openModal(hit);
  } catch {
    document.getElementById("featured").replaceChildren(el("p", { class: "muted", text: "프로젝트 목록을 불러오지 못했습니다." }));
  }
}

main();
