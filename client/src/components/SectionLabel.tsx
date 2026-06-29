interface SectionLabelProps {
  title: string;
  hint?: string;
}

export function SectionLabel({ title, hint }: SectionLabelProps) {
  return (
    <div className="mb-3">
      <h2 className="text-sm font-semibold tracking-tight text-foreground">{title}</h2>
      {hint && <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{hint}</p>}
    </div>
  );
}
