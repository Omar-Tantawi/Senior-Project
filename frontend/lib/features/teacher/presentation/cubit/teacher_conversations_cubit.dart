import 'dart:async';

import 'package:equatable/equatable.dart';
import 'package:first_try/features/chat/data/models/chat_models.dart';
import 'package:first_try/features/teacher/data/models/teacher_extra_models.dart';
import 'package:first_try/features/teacher/data/repos/teacher_repo.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

// ── States ───────────────────────────────────────────────────────────────────

sealed class TeacherConversationsState extends Equatable {
  const TeacherConversationsState();
  @override
  List<Object?> get props => [];
}

class TeacherConversationsInitial extends TeacherConversationsState {
  const TeacherConversationsInitial();
}

class TeacherConversationsLoading extends TeacherConversationsState {
  const TeacherConversationsLoading();
}

class TeacherConversationsLoaded extends TeacherConversationsState {
  final List<ConversationModel> conversations;

  /// Parents list — loaded lazily when the "New Chat" sheet opens.
  final List<ParentSummaryModel> parents;
  final bool parentsLoading;

  /// While a new conversation is being created.
  final bool starting;

  const TeacherConversationsLoaded({
    this.conversations = const [],
    this.parents = const [],
    this.parentsLoading = false,
    this.starting = false,
  });

  TeacherConversationsLoaded copyWith({
    List<ConversationModel>? conversations,
    List<ParentSummaryModel>? parents,
    bool? parentsLoading,
    bool? starting,
  }) =>
      TeacherConversationsLoaded(
        conversations:  conversations  ?? this.conversations,
        parents:        parents        ?? this.parents,
        parentsLoading: parentsLoading ?? this.parentsLoading,
        starting:       starting       ?? this.starting,
      );

  @override
  List<Object?> get props => [conversations, parents, parentsLoading, starting];
}

class TeacherConversationsError extends TeacherConversationsState {
  final String message;
  const TeacherConversationsError(this.message);
  @override
  List<Object?> get props => [message];
}

// ── Cubit ────────────────────────────────────────────────────────────────────

class TeacherConversationsCubit extends Cubit<TeacherConversationsState> {
  final TeacherRepo _repo;

  /// Exposed so screens can create a TeacherChatCubit with the same repo.
  TeacherRepo get repo => _repo;

  TeacherConversationsCubit({required TeacherRepo repo})
      : _repo = repo,
        super(const TeacherConversationsInitial());

  Future<void> load() async {
    emit(const TeacherConversationsLoading());
    try {
      final conversations = await _repo.getConversations();
      emit(TeacherConversationsLoaded(conversations: conversations));
    } catch (e) {
      emit(TeacherConversationsError(e.toString()));
    }
  }

  Future<void> refresh() async {
    try {
      final conversations = await _repo.getConversations();
      final s = state;
      final base = s is TeacherConversationsLoaded ? s : const TeacherConversationsLoaded();
      emit(base.copyWith(conversations: conversations));
    } catch (_) {}
  }

  /// Lazy-loads parents for the "New Chat" picker.
  Future<void> loadParents() async {
    final s = state;
    if (s is! TeacherConversationsLoaded) return;
    if (s.parents.isNotEmpty || s.parentsLoading) return;

    emit(s.copyWith(parentsLoading: true));
    try {
      final parents = await _repo.getParents();
      final cur = state;
      if (cur is TeacherConversationsLoaded) {
        emit(cur.copyWith(parents: parents, parentsLoading: false));
      }
    } catch (_) {
      final cur = state;
      if (cur is TeacherConversationsLoaded) {
        emit(cur.copyWith(parentsLoading: false));
      }
    }
  }

  /// Starts (or reopens) a conversation and returns the conversation id.
  Future<int?> startConversation({
    required int parentUserId,
    required String body,
  }) async {
    final s = state;
    final base = s is TeacherConversationsLoaded ? s : const TeacherConversationsLoaded();
    emit(base.copyWith(starting: true));
    try {
      final result = await _repo.startConversation(
        parentUserId: parentUserId,
        body: body,
      );
      final convId = (result['conversation_id'] as num).toInt();
      // Refresh list so the new conversation appears.
      final updated = await _repo.getConversations();
      final cur = state;
      if (cur is TeacherConversationsLoaded) {
        emit(cur.copyWith(conversations: updated, starting: false));
      }
      return convId;
    } catch (e) {
      final cur = state;
      if (cur is TeacherConversationsLoaded) {
        emit(cur.copyWith(starting: false));
      }
      emit(TeacherConversationsError(e.toString()));
      return null;
    }
  }
}
