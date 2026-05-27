import 'package:equatable/equatable.dart';
import 'package:first_try/features/chat/data/models/chat_models.dart';
import 'package:first_try/features/parent/data/models/parent_extra_models.dart';
import 'package:first_try/features/parent/data/repos/parent_repo.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

// ── States ───────────────────────────────────────────────────────────────────

sealed class ParentConversationsState extends Equatable {
  const ParentConversationsState();
  @override
  List<Object?> get props => [];
}

class ParentConversationsInitial extends ParentConversationsState {
  const ParentConversationsInitial();
}

class ParentConversationsLoading extends ParentConversationsState {
  const ParentConversationsLoading();
}

class ParentConversationsLoaded extends ParentConversationsState {
  final List<ConversationModel> conversations;

  /// Teachers list — loaded lazily when the "New Chat" sheet opens.
  final List<TeacherSummaryModel> teachers;
  final bool teachersLoading;

  final bool starting;

  const ParentConversationsLoaded({
    this.conversations  = const [],
    this.teachers       = const [],
    this.teachersLoading = false,
    this.starting       = false,
  });

  ParentConversationsLoaded copyWith({
    List<ConversationModel>? conversations,
    List<TeacherSummaryModel>? teachers,
    bool? teachersLoading,
    bool? starting,
  }) =>
      ParentConversationsLoaded(
        conversations:  conversations  ?? this.conversations,
        teachers:       teachers       ?? this.teachers,
        teachersLoading: teachersLoading ?? this.teachersLoading,
        starting:       starting       ?? this.starting,
      );

  @override
  List<Object?> get props => [conversations, teachers, teachersLoading, starting];
}

class ParentConversationsError extends ParentConversationsState {
  final String message;
  const ParentConversationsError(this.message);
  @override
  List<Object?> get props => [message];
}

// ── Cubit ────────────────────────────────────────────────────────────────────

class ParentConversationsCubit extends Cubit<ParentConversationsState> {
  final ParentRepo _repo;

  /// Exposed so screens can create a ParentChatCubit with the same repo.
  ParentRepo get repo => _repo;

  ParentConversationsCubit({required ParentRepo repo})
      : _repo = repo,
        super(const ParentConversationsInitial());

  Future<void> load() async {
    emit(const ParentConversationsLoading());
    try {
      final conversations = await _repo.getConversations();
      emit(ParentConversationsLoaded(conversations: conversations));
    } catch (e) {
      emit(ParentConversationsError(e.toString()));
    }
  }

  Future<void> refresh() async {
    try {
      final conversations = await _repo.getConversations();
      final s = state;
      final base = s is ParentConversationsLoaded ? s : const ParentConversationsLoaded();
      emit(base.copyWith(conversations: conversations));
    } catch (_) {}
  }

  /// Lazy-loads teachers for the "New Chat" picker.
  Future<void> loadTeachers() async {
    final s = state;
    if (s is! ParentConversationsLoaded) return;
    if (s.teachers.isNotEmpty || s.teachersLoading) return;

    emit(s.copyWith(teachersLoading: true));
    try {
      final teachers = await _repo.getTeachers();
      final cur = state;
      if (cur is ParentConversationsLoaded) {
        emit(cur.copyWith(teachers: teachers, teachersLoading: false));
      }
    } catch (_) {
      final cur = state;
      if (cur is ParentConversationsLoaded) {
        emit(cur.copyWith(teachersLoading: false));
      }
    }
  }

  /// Starts (or reopens) a conversation and returns the conversation id.
  Future<int?> startConversation({
    required int teacherId, // Teacher model id
    required String body,
  }) async {
    final s = state;
    final base = s is ParentConversationsLoaded ? s : const ParentConversationsLoaded();
    emit(base.copyWith(starting: true));
    try {
      final result = await _repo.startConversation(
        teacherId: teacherId,
        body: body,
      );
      final convId = (result['conversation_id'] as num).toInt();
      final updated = await _repo.getConversations();
      final cur = state;
      if (cur is ParentConversationsLoaded) {
        emit(cur.copyWith(conversations: updated, starting: false));
      }
      return convId;
    } catch (e) {
      final cur = state;
      if (cur is ParentConversationsLoaded) {
        emit(cur.copyWith(starting: false));
      }
      emit(ParentConversationsError(e.toString()));
      return null;
    }
  }
}
