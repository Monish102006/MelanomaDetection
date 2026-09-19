import { useRef, useState } from "react";
import { motion } from "framer-motion";
import { ImagePlus, RotateCcw, Upload, X } from "lucide-react";
import { ActionButton } from "./primitives";
import { cn } from "@/lib/utils";

export function UploadPanel({
  previewUrl,
  fileName,
  running,
  onFile,
  onStart,
  onReset,
}: {
  previewUrl: string | null;
  fileName?: string;
  running: boolean;
  onFile: (f: File | null) => void;
  onStart: () => void;
  onReset: () => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const accept = (files: FileList | null) => {
    const f = files?.[0];
    if (f && f.type.startsWith("image/")) onFile(f);
  };

  return (
    <div className="glass rounded-2xl p-6">
      <div className="flex items-center justify-between">
        <span className="label-mono">Input · lesion image</span>
        {previewUrl ? (
          <button
            onClick={onReset}
            aria-label="Clear image"
            className="text-muted-foreground transition-colors hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        ) : null}
      </div>

      <div
        role="button"
        tabIndex={0}
        aria-label="Upload lesion image"
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          accept(e.dataTransfer.files);
        }}
        className={cn(
          "mt-4 flex min-h-[280px] cursor-pointer flex-col items-center justify-center overflow-hidden rounded-xl border border-dashed p-6 text-center transition-all duration-300",
          dragging ? "border-primary bg-primary/10" : "border-border bg-secondary/25 hover:border-primary/50",
        )}
      >
        {previewUrl ? (
          <motion.img
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            src={previewUrl}
            alt="Uploaded lesion preview"
            className="max-h-[260px] w-auto rounded-lg object-contain"
          />
        ) : (
          <>
            <span className="flex h-12 w-12 items-center justify-center rounded-full border border-primary/35 bg-primary/10">
              <ImagePlus className="h-5 w-5 text-primary" />
            </span>
            <p className="mt-4 text-sm text-foreground">Drop a lesion image here</p>
            <p className="mt-1 text-xs text-muted-foreground">
              or click to browse · PNG, JPG · dermoscopic or clinical
            </p>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="sr-only"
          onChange={(e) => accept(e.target.files)}
        />
      </div>

      {fileName ? (
        <p className="mt-3 truncate font-mono text-xs text-muted-foreground">{fileName}</p>
      ) : null}

      <div className="mt-5 flex flex-wrap gap-3">
        <ActionButton onClick={onStart} disabled={!previewUrl || running}>
          <Upload className="h-4 w-4" />
          {running ? "Analyzing…" : "Start analysis"}
        </ActionButton>
        <ActionButton variant="ghost" onClick={onReset} disabled={running}>
          <RotateCcw className="h-4 w-4" />
          Reset
        </ActionButton>
      </div>

      <p className="mt-4 text-xs leading-relaxed text-muted-foreground">
        Images are processed for research demonstration only. Do not upload identifiable patient
        data.
      </p>
    </div>
  );
}
