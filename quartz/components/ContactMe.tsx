import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

const ContactMe: QuartzComponent = (_props: QuartzComponentProps) => {
  return (
    <div class="contact-me">
      <div class="contact-title">Contact me</div>

      <div class="contact-links">
        <a
          href="mailto:your-email@example.com"
          class="contact-link"
          aria-label="Email"
          title="Email"
        >
          ✉️
        </a>
        <a
          href="https://github.com/Yun-sooyong"
          class="contact-link"
          aria-label="GitHub"
          title="GitHub"
          target="_blank"
          rel="noopener noreferrer"
        >
          🤖
        </a>
      </div>
    </div>
  )
}

ContactMe.css = `
.contact-me {
  margin-top: 1rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--lightgray);
}

.contact-title {
  font-size: 0.9rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
}

.contact-links {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.contact-link {
  text-decoration: none;
  font-size: 1.1rem;
}
`

export default (() => ContactMe) satisfies QuartzComponentConstructor