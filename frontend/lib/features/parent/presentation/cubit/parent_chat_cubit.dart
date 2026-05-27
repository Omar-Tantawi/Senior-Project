import 'dart:async';

import 'package:equatable/equatable.dart';
import 'package:first_try/features/chat/data/models/chat_models.dart';
import 'package:first_try/features/parent/data/repos/parent_repo.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

// ── States ───────────────────────────────────────────────────────────────────

sealed class ParentChatState extends Equatable {
  const ParentChatState();
  @override
  List<Object?> get props => [];
}

class ParentChatLoading extends ParentChatState {
  const ParentChatLoading();
}

class ParentChatLoaded extends ParentChatState {
  final List<ChatMessageModel> messages;
  final bool sending;

  const ParentChatLoaded({
    this.messages = const [],
    this.sending  = false,
  });

  ParentChatLoaded copyWith({
    List<ChatMessageModel>? messages,
    bool? sending,
  }) =>
      ParentChatLoaded(
        messages: messages ?? this.messages,
        sending:  sending  ?? this.sending,
      );

  @override
  List<Object?> get props => [messages, sending];
}

class ParentChatError extends ParentChatState {
  final String message;
  const ParentChatError(this.message);
  @override
  List<Object?> get props => [message];
}

// ── Cubit ────────────────────────────────────────────────────────────────────

class ParentChatCubit extends Cubit<ParentChatState> {
  final ParentRepo _repo;
  final int conversationId;

  Timer? _pollTimer;

  ParentChatCubit({
    required ParentRepo repo,
    required this.conversationId,
  })  : _repo = repo,
        super(const ParentChatLoading());

  Future<void> load() async {
    try {
      final messages = await _repo.getConversationMessages(conversationId);
      emit(ParentChatLoaded(messages: messages));
      _startPolling();
    } catch (e) {
      emit(ParentChatError(e.toString()));
    }
  }

  void _startPolling() {
    _pollTimer?.cancel();
    _pollTimer = Timer.periodic(const Duration(seconds: 5), (_) => _poll());
  }

  Future<void> _poll() async {
    final s = state;
    if (s is! ParentChatLoaded) return;
    try {
      final lastId = s.messages.isEmpty ? null : s.messages.last.id;
      final fresh = await _repo.getConversationMessages(
        conversationId,
        afterId: lastId,
      );
      if (fresh.isNotEmpty) {
        emit(s.copyWith(messages: [...s.messages, ...fresh]));
      }
    } catch (_) {}
  }

  Future<bool> sendReply(String body) async {
    final s = state;
    final base = s is ParentChatLoaded ? s : const ParentChatLoaded();
    emit(base.copyWith(sending: true));
    try {
      final msg = await _repo.sendConversationReply(conversationId, body);
      emit(base.copyWith(
        messages: [...base.messages, msg],
        sending: false,
      ));
      return true;
    } catch (e) {
      emit(base.copyWith(sending: false));
      return false;
    }
  }

  @override
  Future<void> close() {
    _pollTimer?.cancel();
    return super.close();
  }
}
