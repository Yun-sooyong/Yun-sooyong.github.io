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
  width: 100%;
  margin-top: 2rem;
  padding-top: 1rem;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.contact-heading {
  width: 100%;
  text-align: center;
  font-size: 1rem;
  font-weight: 600;
  color: var(--gray);
  margin-bottom: 0.75rem;
  letter-spacing: 0.03em;
}

.contact-links {
  width: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 0.45rem;
  font-size: 0.95rem;
}

.contact-link {
  text-decoration: none;
  color: var(--secondary);
}

.contact-link:hover {
  text-decoration: underline;
}

.contact-separator {
  color: var(--gray);
}
`

export default (() => ContactMe) satisfies QuartzComponentConstructor