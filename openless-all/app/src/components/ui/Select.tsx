import { Select as BaseSelect } from '@base-ui/react/select';
import { Icon } from '../Icon';

interface SelectProps {
  value: string;
  onChange: (value: string) => void;
  options: readonly { value: string; label: string }[];
  ariaLabel: string;
  disabled?: boolean;
}

/** Standard Base UI Select; keyboard navigation, typeahead and focus return stay upstream. */
export function Select({ value, onChange, options, ariaLabel, disabled }: SelectProps) {
  return (
    <BaseSelect.Root
      value={value}
      items={options}
      disabled={disabled}
      onValueChange={(next) => {
        if (next !== null) onChange(next);
      }}
    >
      <BaseSelect.Trigger className="ol-select-trigger" aria-label={ariaLabel}>
        <BaseSelect.Value className="ol-select-value" />
        <BaseSelect.Icon>
          <Icon name="chevDown" size={11} />
        </BaseSelect.Icon>
      </BaseSelect.Trigger>
      <BaseSelect.Portal className="ol-popup-portal">
        <BaseSelect.Positioner
          className="ol-select-positioner"
          positionMethod="fixed"
          sideOffset={4}
          collisionPadding={8}
          alignItemWithTrigger={false}
        >
          <BaseSelect.Popup className="ol-select-popup">
            <BaseSelect.List className="ol-select-list">
              {options.map((option) => (
                <BaseSelect.Item key={option.value} value={option.value} className="ol-select-item">
                  <BaseSelect.ItemText>{option.label}</BaseSelect.ItemText>
                  <BaseSelect.ItemIndicator className="ol-select-check">
                    <Icon name="check" size={12} />
                  </BaseSelect.ItemIndicator>
                </BaseSelect.Item>
              ))}
            </BaseSelect.List>
          </BaseSelect.Popup>
        </BaseSelect.Positioner>
      </BaseSelect.Portal>
    </BaseSelect.Root>
  );
}
