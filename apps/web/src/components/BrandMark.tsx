import Image from 'next/image';

interface BrandMarkProps {
  size?: number;
  showWordmark?: boolean;
  className?: string;
}

export function BrandMark({ size = 28, showWordmark = false, className = '' }: BrandMarkProps) {
  return (
    <span className={`inline-flex items-center gap-2.5 select-none ${className}`.trim()}>
      <Image
        src="/brand/arkhiverag-icon.png"
        alt="ArkhiveRAG"
        width={size}
        height={size}
        className="shrink-0 dark:hidden"
        priority={size >= 36}
      />
      <Image
        src="/brand/arkhiverag-icon-dark.png"
        alt="ArkhiveRAG"
        width={size}
        height={size}
        className="hidden shrink-0 dark:block"
        priority={size >= 36}
      />
      {showWordmark && (
        <span className="text-[15px] font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
          Arkhive<span className="text-[#00A6CE]">RAG</span>
        </span>
      )}
    </span>
  );
}
