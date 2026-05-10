import { useEffect, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useLocation } from 'react-router-dom';
import { getAgentModels, getAgentStatus, getModels, selectAgentModel, selectModel } from '../../api/client';
import { useStatus } from '../../hooks/useStatus';

const CONVERSATIONS_STORAGE_KEY = 'rca-conversations';
const CHAT_MODEL_STORAGE_KEY = 'rca-current-chat-model';

type ChatLoadingEventDetail = {
  isLoading: boolean;
};

type ChatActiveModelEventDetail = {
  model: string;
};

export function StatusBar() {
  const location = useLocation();
  const isAgentRoute = location.pathname === '/agent';
  const isLibraryRoute = location.pathname === '/library';
  const isChatRoute = location.pathname === '/';
  const showModelPicker = isChatRoute || isAgentRoute;
  const queryClient = useQueryClient();
  const { data, isLoading } = useStatus();
  const [isPickerOpen, setPickerOpen] = useState(false);
  const [selectedModelOverride, setSelectedModelOverride] = useState<string | null>(null);
  const [chatIsLoading, setChatIsLoading] = useState(false);
  const pickerRef = useRef<HTMLDivElement | null>(null);

  const modelsQuery = useQuery({
    queryKey: ['models'],
    queryFn: getModels,
    staleTime: 30_000,
  });

  const agentModelsQuery = useQuery({
    queryKey: ['agent-models'],
    queryFn: getAgentModels,
    staleTime: 30_000,
    enabled: isAgentRoute,
  });

  const agentStatusQuery = useQuery({
    queryKey: ['agent-status'],
    queryFn: getAgentStatus,
    staleTime: 15_000,
    enabled: isAgentRoute,
  });

  const selectMutation = useMutation({
    mutationFn: (model: string) => (isAgentRoute ? selectAgentModel(model) : selectModel(model)),
    onSuccess: (result) => {
      setSelectedModelOverride(result.model);
      if (isChatRoute) {
        window.localStorage.setItem(CHAT_MODEL_STORAGE_KEY, result.model);
        if (isPickerOpen) {
          window.dispatchEvent(
            new CustomEvent('rca-chat-model-selected', {
              detail: { model: result.model },
            }),
          );
        }
      }
      void queryClient.invalidateQueries({ queryKey: ['status'] });
      void queryClient.invalidateQueries({ queryKey: ['models'] });
      void queryClient.invalidateQueries({ queryKey: ['agent-models'] });
      void queryClient.invalidateQueries({ queryKey: ['agent-status'] });
      setPickerOpen(false);
    },
  });

  useEffect(() => {
    setSelectedModelOverride(null);
    setPickerOpen(false);
  }, [isAgentRoute, isLibraryRoute, showModelPicker]);

  useEffect(() => {
    const handleChatLoading = (event: Event) => {
      const detail = (event as CustomEvent<ChatLoadingEventDetail>).detail;
      if (typeof detail?.isLoading === 'boolean') {
        setChatIsLoading(detail.isLoading);
      }
    };

    window.addEventListener('rca-chat-loading', handleChatLoading as EventListener);
    return () =>
      window.removeEventListener('rca-chat-loading', handleChatLoading as EventListener);
  }, []);

  useEffect(() => {
    if (!isPickerOpen) {
      return;
    }

    const handleOutsideClick = (event: MouseEvent) => {
      if (pickerRef.current && !pickerRef.current.contains(event.target as Node)) {
        setPickerOpen(false);
      }
    };

    window.addEventListener('mousedown', handleOutsideClick);
    return () => window.removeEventListener('mousedown', handleOutsideClick);
  }, [isPickerOpen]);

  const model = isLoading ? '--' : (data?.model ?? '--');
  const backend = isLoading ? '--' : (data?.backend ?? '--');
  const ollamaConnected = isLoading ? null : Boolean(data?.ollama_connected);
  const capableSet = new Set(agentModelsQuery.data?.models ?? []);
  const recommended = isAgentRoute
    ? (modelsQuery.data?.recommended ?? []).filter((option) => capableSet.has(option.name))
    : (modelsQuery.data?.recommended ?? []);
  const other = isAgentRoute
    ? (modelsQuery.data?.other ?? []).filter((option) => capableSet.has(option.name))
    : (modelsQuery.data?.other ?? []);
  const modelChoices = [...recommended.map((option) => option.name), ...other.map((option) => option.name)];
  const currentModel = isAgentRoute
    ? selectedModelOverride ?? agentModelsQuery.data?.current ?? agentStatusQuery.data?.model ?? '--'
    : selectedModelOverride ?? modelsQuery.data?.current ?? model;

  const isPickerDisabled = selectMutation.isPending || (isChatRoute && chatIsLoading);

  useEffect(() => {
    if (!isChatRoute) {
      return;
    }

    const handleActiveModel = (event: Event) => {
      const detail = (event as CustomEvent<ChatActiveModelEventDetail>).detail;
      const nextModel = detail?.model;
      if (!nextModel) {
        return;
      }

      setSelectedModelOverride(nextModel);
      if (nextModel !== currentModel && !selectMutation.isPending) {
        selectMutation.mutate(nextModel);
      }
    };

    window.addEventListener('rca-chat-active-model', handleActiveModel as EventListener);
    return () =>
      window.removeEventListener('rca-chat-active-model', handleActiveModel as EventListener);
  }, [currentModel, isChatRoute, selectMutation.isPending]);

  useEffect(() => {
    if (!isChatRoute) {
      return;
    }

    const syncConversationModel = () => {
      const raw = window.localStorage.getItem(CONVERSATIONS_STORAGE_KEY);
      if (!raw) {
        return;
      }

      try {
        const parsed = JSON.parse(raw) as {
          activeConversationId?: string;
          conversations?: Array<{ id: string; model?: string }>;
        };

        const activeId = parsed.activeConversationId;
        const conversations = parsed.conversations ?? [];
        const active = conversations.find((conversation) => conversation.id === activeId);
        const modelFromConversation = active?.model;
        if (!modelFromConversation) {
          return;
        }

        if (modelFromConversation !== currentModel && !selectMutation.isPending) {
          selectMutation.mutate(modelFromConversation);
        }

        window.localStorage.setItem(CHAT_MODEL_STORAGE_KEY, modelFromConversation);
      } catch {
        // Ignore malformed local state.
      }
    };

    syncConversationModel();
    window.addEventListener('rca-conversations-updated', syncConversationModel as EventListener);
    return () =>
      window.removeEventListener('rca-conversations-updated', syncConversationModel as EventListener);
  }, [currentModel, isChatRoute, selectMutation.isPending]);

  useEffect(() => {
    if (isChatRoute && currentModel && currentModel !== '--') {
      window.localStorage.setItem(CHAT_MODEL_STORAGE_KEY, currentModel);
    }
  }, [currentModel, isChatRoute]);

  const rightLabel =
    ollamaConnected === null
      ? '○ Ollama —'
      : ollamaConnected
        ? '● Ollama connected'
        : '○ Ollama offline';

  return (
    <footer className="status-bar">
      <div className="status-left" ref={pickerRef}>
        <span>RCA</span>
        <span>·</span>
        {showModelPicker ? (
          <>
            <button
              type="button"
              className="status-model-button"
              onClick={() => {
                if (!isPickerDisabled) {
                  setPickerOpen((open) => !open);
                }
              }}
              aria-expanded={isPickerOpen}
              aria-label="Choose model"
              disabled={isPickerDisabled}
            >
              {currentModel}
            </button>
            <span>·</span>
          </>
        ) : null}
        <span>{backend}</span>

        {isPickerOpen && showModelPicker ? (
          <div className="status-model-picker">
            {modelChoices.length === 0 ? (
              <div className="status-model-empty">No models found</div>
            ) : (
              <>
                {recommended.length > 0 ? (
                  <>
                    <div className="status-model-group-label">Recommended for RCA</div>
                    {recommended.map((option) => (
                      <button
                        key={option.name}
                        type="button"
                        className={`status-model-option ${option.name === currentModel ? 'is-active' : ''}`}
                        onClick={() => selectMutation.mutate(option.name)}
                        disabled={isPickerDisabled}
                        title={option.reasons.join(', ')}
                      >
                        {option.name}
                      </button>
                    ))}
                  </>
                ) : null}

                {other.length > 0 ? (
                  <>
                    <div className="status-model-group-label">Other compatible</div>
                    {other.map((option) => (
                      <button
                        key={option.name}
                        type="button"
                        className={`status-model-option ${option.name === currentModel ? 'is-active' : ''}`}
                        onClick={() => selectMutation.mutate(option.name)}
                        disabled={isPickerDisabled}
                        title={option.reasons.join(', ')}
                      >
                        {option.name}
                      </button>
                    ))}
                  </>
                ) : null}
              </>
            )}
          </div>
        ) : null}
      </div>
      <span
        className={`status-right ${
          ollamaConnected === null
            ? 'status-unknown'
            : ollamaConnected
              ? 'status-online'
              : 'status-offline'
        }`}
      >
        {rightLabel}
      </span>
    </footer>
  );
}
