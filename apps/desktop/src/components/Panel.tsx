import type { ReactNode } from "react";

export function Panel({
  title,
  kicker,
  aside,
  children,
  className = "",
  bodyClassName = "",
  flush = false,
}: {
  title: string;
  kicker?: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
  flush?: boolean;
}) {
  return (
    <section className={`panel ${className}`.trim()}>
      <header className="panel-head">
        <div className="panel-title-wrap">
          <h2 className="panel-title">{title}</h2>
          {kicker ? <span className="panel-kicker">{kicker}</span> : null}
        </div>
        {aside ? <div className="panel-aside">{aside}</div> : null}
      </header>
      <div className={`panel-body ${flush ? "panel-body-flush" : ""} ${bodyClassName}`.trim()}>{children}</div>
    </section>
  );
}

export function CardRule({ children }: { children: ReactNode }) {
  return <div className="card-rule">{children}</div>;
}

export function EmptyState({
  title,
  body,
  action,
  compact = false,
}: {
  title: string;
  body: string;
  action?: ReactNode;
  compact?: boolean;
}) {
  return (
    <div className={`empty-state ${compact ? "empty-state-compact" : ""}`.trim()}>
      <p className="empty-title">{title}</p>
      <p className="empty-body">{body}</p>
      {action ? <div className="empty-action">{action}</div> : null}
    </div>
  );
}
