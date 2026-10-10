const tabs = [...document.querySelectorAll('[role="tab"]')];
function selectTab(tab) {
  tabs.forEach((item) => {
    const selected = item === tab;
    item.setAttribute("aria-selected", String(selected));
    item.tabIndex = selected ? 0 : -1;
    document.getElementById(item.getAttribute("aria-controls")).hidden =
      !selected;
  });
}
tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => selectTab(tab));
  tab.addEventListener("keydown", (event) => {
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    if (event.key === "ArrowLeft")
      next = (index + tabs.length - 1) % tabs.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = tabs.length - 1;
    if (next === undefined) return;
    event.preventDefault();
    selectTab(tabs[next]);
    tabs[next].focus();
  });
});
const dialog = document.getElementById("figure-dialog");
const figureImage = dialog.querySelector("img");
const figureTitle = document.getElementById("figure-title");
const stage = dialog.querySelector(".zoom-stage");
const pdfLink = dialog.querySelector(".figure-pdf");
let figureTrigger;
let zoom = "fit";
function setZoom(value) {
  zoom = value;
  figureImage.style.width =
    value === "fit" ? "100%" : figureImage.naturalWidth * Number(value) + "px";
  dialog
    .querySelectorAll("[data-zoom]")
    .forEach((button) =>
      button.setAttribute(
        "aria-pressed",
        String(button.dataset.zoom === value),
      ),
    );
}
figureImage.addEventListener("load", () => setZoom(zoom));
document.querySelectorAll("[data-figure]").forEach((button) => {
  button.addEventListener("click", () => {
    figureTrigger = button;
    figureTitle.textContent = button.dataset.caption || "Research figure";
    figureImage.alt = button.querySelector("img").alt;
    figureImage.src = button.dataset.figure;
    pdfLink.hidden = !button.dataset.pdf;
    if (button.dataset.pdf) pdfLink.href = button.dataset.pdf;
    else pdfLink.removeAttribute("href");
    setZoom("fit");
    dialog.showModal();
    stage.scrollTo(0, 0);
  });
});
dialog
  .querySelectorAll("[data-zoom]")
  .forEach((button) =>
    button.addEventListener("click", () => setZoom(button.dataset.zoom)),
  );
dialog
  .querySelector(".figure-close")
  .addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (event) => {
  if (event.target !== dialog) return;
  const bounds = dialog.getBoundingClientRect();
  if (
    event.clientX < bounds.left ||
    event.clientX > bounds.right ||
    event.clientY < bounds.top ||
    event.clientY > bounds.bottom
  )
    dialog.close();
});
dialog.addEventListener("close", () => figureTrigger?.focus());
const navLinks = [...document.querySelectorAll("nav a")];
const observedSections = navLinks
  .map((link) => document.querySelector(link.getAttribute("href")))
  .filter(Boolean);
if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        navLinks.forEach((link) => {
          const active = link.getAttribute("href") === "#" + entry.target.id;
          link.classList.toggle("active", active);
          if (active) link.setAttribute("aria-current", "location");
          else link.removeAttribute("aria-current");
        });
      });
    },
    { rootMargin: "-15% 0px -60% 0px" },
  );
  observedSections.forEach((section) => observer.observe(section));
}
