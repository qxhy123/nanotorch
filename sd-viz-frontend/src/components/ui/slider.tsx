import React from 'react';

export interface SliderProps extends React.InputHTMLAttributes<HTMLInputElement> {
  value: number;
  onValueChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  label?: string;
}

export const Slider = ({ value, onValueChange, min = 0, max = 100, step = 1, label, ...props }: SliderProps) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onValueChange(parseFloat(e.target.value));
  };

  return (
    <div className="space-y-2">
      {label && <label className="text-sm font-medium">{label}: {value}</label>}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={handleChange}
        className="w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
        {...props}
      />
    </div>
  );
};
