import 'dart:async';

import 'package:equatable/equatable.dart';
import 'package:first_try/features/chat/data/models/chat_models.dart';
import 'package:first_try/features/teacher/data/repos/teacher_repo.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

// ── States ───────────────────────────────────────────────────────────────────

sealed class TeacherChatState extends Equatable {
  const TeacherChatState();
  @override
  List<Object?> get props => [];
}

class TeacherChatLoading extends TeacherChatState {
  const TeacherChatLoading();
}

class TeacherChatLoaded extends TeacherChatState {
  final List<ChatMessageModel> messages;
  final bool sending;

  const TeacherChatLoaded({
    this.messages = const [],
    this.sending  = false,
  });

  TeacherChatLoaded copyWith({
    List<ChatMessageModel>? messages,
    bool? sending,
  }) =>
      TeacherChatLoaded(
        messages: messages ?? this.messages,
        sending:  sending  ?? this.sending,
      );

  @override
  List<Object?> get props => [messages, sending];
}

class TeacherChatError extends TeacherChatState {
  final String message;
  const TeacherChatError(this.message);
  @override
  List<Object?> get props => [message];
}

// ── Cubit ────────────────────────────────────────────────────────────────────

class TeacherChatCubit extends Cubit<TeacherChatState> {
  final TeacherRepo _repo;
  final int conversationId;

  Timer? _pollTimer;

  TeacherChatCubit({
    required TeacherRepo repo,
    required this.conversationId,
  })  : _repo = repo,
        super(const TeacherChatLoading());

  // ── Load (initial) ────────────────────────────────────────────────────────

  Future<void> load() async {
    try {
      final messages = await _repo.getConversationMessages(conversationId);
      emit(TeacherChatLoaded(messages: messages));
      _startPolling();
    } catch (e) {
      emit(TeacherChatError(e.toString()));
    }
  }

  // ── Polling ───────────────────────────────────────────────────────────────

  void _startPolling() {
    _pollTimer?.cancel();
    _pollTimer = Timer.periodic(const Duration(seconds: 5), (_) => _poll());
  }

  Future<void> _poll() async {
    final s = state;
    if (s is! TeacherChatLoaded) return;
    try {
      final lastId = s.messages.isEmpty ? null : s.messages.last.id;
      final fresh = await _repo.getConversationMessages(
        conversationId,
        afterId: lastId,
      );
      if (fresh.isNotEmpty) {
        emit(s.copyWith(messages: [...s.messages, ...fresh]));
      }
    } catch (_) {
      // Swallow poll errors silently — don't disrupt the UI.
    }
  }

  // ── Send reply ────────────────────────────────────────────────────────────

  Future<bool> sendReply(String body) async {
    final s = state;
    final base = s is TeacherChatLoaded ? s : const TeacherChatLoaded();
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

  // ── Cleanup ───────────────────────────────────────────────────────────────

  @override
  Future<void> close() {
    _pollTimer?.cancel();
    return super.close();
  }
}
