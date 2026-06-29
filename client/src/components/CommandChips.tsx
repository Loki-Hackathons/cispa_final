import type { CommandChip } from "../types/status";

interface CommandChipsProps {
  commands: CommandChip[];
}

export function CommandChips({ commands }: CommandChipsProps) {
  if (commands.length === 0) return null;

  const copy = async (command: string) => {
    try {
      await navigator.clipboard.writeText(command);
    } catch {
      /* ignore */
    }
  };

  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      {commands.map((chip) => (
        <button
          key={`${chip.label}-${chip.command}`}
          type="button"
          onClick={() => void copy(chip.command)}
          title={chip.command}
          className="rounded border border-border bg-muted/50 px-2 py-0.5 font-mono text-[10px] text-muted-foreground transition-colors hover:border-primary/40 hover:text-primary"
        >
          {chip.label}
        </button>
      ))}
    </div>
  );
}
