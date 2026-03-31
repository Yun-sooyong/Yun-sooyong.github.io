import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

const ContactMe: QuartzComponent = (_props: QuartzComponentProps) => {
  return (
    <div class="contact-me">
      <div class="contact-divider" />
      <div class="contact-title">- Contact Me -</div>
      <div class="contact-divider" />

      <div class="contact-links">
        <a
          href="mailto:ysyg22@gmail.com"
          class="contact-link"
          aria-label="Email"
          title="Email"
        >
          #Email
        </a>
        <a
          href="https://github.com/Yun-sooyong"
          class="contact-link"
          aria-label="GitHub"
          title="GitHub"
          target="_blank"
          rel="noopener noreferrer"
        >
          #GitHub
        </a>
      </div>
    </div>
  )
}

ContactMe.css = `
.contact-me {
  margin-top: 2rem;
  padding-top: 1rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}

.contact-divider {
  width: 100%;
  border-top: 1px solid var(--lightgray);
  margin: 0.3rem 0;
}

.contact-title {
  font-size: 0.85rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--gray);
}

.contact-links {
  margin-top: 0.8rem;
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 1rem;
}

.contact-link {
  text-decoration: none;
  color: var(--secondary);
  font-size: 0.95rem;
}

.contact-link:hover {
  text-decoration: underline;
}
`

export default (() => ContactMe) satisfies QuartzComponentConstructor