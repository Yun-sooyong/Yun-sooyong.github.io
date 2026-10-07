// data/projects.json을 읽어 프로젝트 카드를 그린다.
// projects.json은 portfolio-hub의 scripts/export_site_data.py가 만든다. 직접 고치지 말 것.

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const child of children) if (child) node.append(child);
  return node;
}

function card(p) {
  const meta = [];
  if (p.ownership === "team") meta.push(p.team_size ? `팀 프로젝트 · ${p.team_size}명` : "팀 프로젝트");
  else meta.push("개인 프로젝트");
  if (p.period) meta.push(p.period);

  const links = el("p", { class: "card-links" }, [
    p.repo && el("a", { href: p.repo, text: "Repository" }),
    p.notion && el("a", { href: p.notion, text: "상세 (Notion)" }),
    p.demo && el("a", { href: p.demo, text: "Demo" }),
  ]);

  return el("article", { class: "card" }, [
    el("h3", { text: p.name }),
    el("p", { class: "meta", text: meta.join(" · ") }),
    el("p", { text: p.one_liner }),
    p.my_role?.length && el("div", {}, [
      el("h4", { text: "맡은 역할" }),
      el("ul", {}, p.my_role.map((r) => el("li", { text: r }))),
    ]),
    p.highlights?.length && el("div", {}, [
      el("h4", { text: "핵심 구현" }),
      el("ul", {}, p.highlights.map((h) => el("li", { text: h }))),
    ]),
    p.stack?.length && el("ul", { class: "tags" }, p.stack.map((s) => el("li", { text: s }))),
    links.childNodes.length ? links : null,
  ]);
}

async function main() {
  const box = document.getElementById("projects");
  try {
    const res = await fetch("data/projects.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(res.status);
    const { projects } = await res.json();
    box.replaceChildren(...(projects.length ? projects.map(card) : [el("p", { class: "muted", text: "준비 중입니다." })]));
  } catch {
    box.replaceChildren(el("p", { class: "muted", text: "프로젝트 목록을 불러오지 못했습니다." }));
  }
}

main();
