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

function projectRow(p) {
  const tags = [
    p.ownership === "team" ? el("span", { class: "tag team", text: `팀 ${p.team_size ?? ""}인`.trim() }) : el("span", { class: "tag", text: "개인" }),
    ...(p.tags ?? []).map((t) => el("span", { class: `tag t-${t.replace(/\s+/g, "-")}`, text: t })),
    p.award && el("span", { class: "tag award", text: "우수상" }),
  ];
  const [start, end] = (p.period ?? "").split("~").map((x) => x && x.trim());
  return el("li", {}, [
    el("button", { class: "proj", type: "button", "aria-haspopup": "dialog", onclick: () => openModal(p) }, [
      el("span", { class: "when" }, [start, end && el("br"), end && `~ ${end}`]),
      el("span", { class: "p-ico", "aria-hidden": "true", text: p.icon ?? "•" }),
      el("span", {}, [
        el("span", { class: "p-name" }, [p.name, p.subtitle && el("small", { text: p.subtitle })]),
        el("span", { class: "tags" }, tags),
      ]),
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
    p.ai_usage?.length && section("AI 활용", [el("ul", {}, p.ai_usage.map((h) => el("li", { text: h })))]),
    p.stack?.length && section("기술 스택", [el("ul", { class: "chips" }, p.stack.map((s) => el("li", { text: s })))]),
    links.length && section("링크", [el("div", { class: "m-links" }, links),
      p.ownership === "team" && el("p", { class: "note", text: "팀 프로젝트입니다. 위 내용은 본인이 담당한 부분이며, 구현에는 AI 도구(Claude Code)를 함께 썼습니다." })]),
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
    document.getElementById("projects").replaceChildren(...projects.map(projectRow));
    const hit = projects.find((p) => `#${p.id}` === location.hash);
    if (hit) openModal(hit);
  } catch {
    document.getElementById("projects").replaceChildren(el("li", { text: "프로젝트 목록을 불러오지 못했습니다." }));
  }
}

main();
