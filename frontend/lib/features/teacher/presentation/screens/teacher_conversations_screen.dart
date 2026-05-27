import 'package:first_try/core/theme/theme.dart';
import 'package:first_try/core/widgets/shared/empty_state.dart';
import 'package:first_try/core/widgets/shared/error_view.dart';
import 'package:first_try/core/widgets/shared/skeletons.dart';
import 'package:first_try/core/widgets/ui/ui.dart';
import 'package:first_try/features/chat/data/models/chat_models.dart';
import 'package:first_try/features/teacher/data/models/teacher_extra_models.dart';
import 'package:first_try/features/teacher/data/repos/teacher_repo.dart';
import 'package:first_try/features/teacher/presentation/cubit/teacher_chat_cubit.dart';
import 'package:first_try/features/teacher/presentation/cubit/teacher_conversations_cubit.dart';
import 'package:first_try/features/teacher/presentation/screens/teacher_chat_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:intl/intl.dart';

class TeacherConversationsScreen extends StatefulWidget {
  const TeacherConversationsScreen({super.key});

  @override
  State<TeacherConversationsScreen> createState() =>
      _TeacherConversationsScreenState();
}

class _TeacherConversationsScreenState
    extends State<TeacherConversationsScreen> {
  @override
  void initState() {
    super.initState();
    context.read<TeacherConversationsCubit>().load();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Messages', style: TextStyle(fontWeight: FontWeight.w700)),
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () => _showNewChat(context),
        icon: const Icon(Icons.chat_rounded),
        label: const Text('New Chat'),
      ),
      body: BlocConsumer<TeacherConversationsCubit, TeacherConversationsState>(
        listener: (context, state) {
          if (state is TeacherConversationsError) {
            ScaffoldMessenger.of(context).showSnackBar(SnackBar(
              content: Text(state.message),
              backgroundColor: Theme.of(context).colorScheme.error,
            ));
          }
        },
        builder: (context, state) {
          if (state is TeacherConversationsInitial ||
              state is TeacherConversationsLoading) {
            return const CardListSkeleton(cardHeight: 72);
          }
          if (state is TeacherConversationsError) {
            return ErrorView(
              message: state.message,
              onRetry: () => context.read<TeacherConversationsCubit>().load(),
            );
          }
          if (state is! TeacherConversationsLoaded) {
            return const SizedBox.shrink();
          }

          if (state.conversations.isEmpty) {
            return const EmptyState(
              icon: Icons.chat_bubble_outline_rounded,
              title: 'No conversations yet',
              subtitle: 'Tap "New Chat" to message a parent.',
            );
          }

          return RefreshIndicator(
            onRefresh: () =>
                context.read<TeacherConversationsCubit>().refresh(),
            child: ListView.separated(
              padding: const EdgeInsets.fromLTRB(0, 8, 0, 100),
              itemCount: state.conversations.length,
              separatorBuilder: (_, __) => const Divider(height: 1, indent: 72),
              itemBuilder: (context, i) => _ConvTile(
                conv: state.conversations[i],
                onTap: () => _openChat(context, state.conversations[i]),
              ),
            ),
          );
        },
      ),
    );
  }

  void _openChat(BuildContext context, ConversationModel conv) {
    final repo = context.read<TeacherConversationsCubit>().state;
    // Get repo from the cubit — pass through via the shell
    final cubit = context.read<TeacherConversationsCubit>();
    // We need the repo to create a chat cubit. Access it via the shell.
    // Use a navigator push with a new BlocProvider.
    final teacherRepo = _getRepo(context);
    if (teacherRepo == null) return;

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => BlocProvider(
          create: (_) => TeacherChatCubit(
            repo: teacherRepo,
            conversationId: conv.id,
          )..load(),
          child: TeacherChatScreen(
            conversationId: conv.id,
            otherUserName: conv.otherUserName,
          ),
        ),
      ),
    ).then((_) => cubit.refresh());
  }

  void _showNewChat(BuildContext context) {
    final cubit = context.read<TeacherConversationsCubit>();
    cubit.loadParents();
    showAppBottomSheet<void>(
      context: context,
      title: 'New Chat',
      subtitle: 'Choose a parent and send a first message.',
      builder: (ctx) => BlocProvider.value(
        value: cubit,
        child: _NewChatSheet(
          onStarted: (convId, otherName) {
            final repo = _getRepo(context);
            if (repo == null) return;
            Navigator.pop(ctx);
            Navigator.push(
              context,
              MaterialPageRoute(
                builder: (_) => BlocProvider(
                  create: (_) => TeacherChatCubit(
                    repo: repo,
                    conversationId: convId,
                  )..load(),
                  child: TeacherChatScreen(
                    conversationId: convId,
                    otherUserName: otherName,
                  ),
                ),
              ),
            ).then((_) => cubit.refresh());
          },
        ),
      ),
    );
  }

  /// Retrieves the TeacherRepo from the shell's cubit.
  TeacherRepo? _getRepo(BuildContext context) {
    try {
      return context.read<TeacherConversationsCubit>().repo;
    } catch (_) {
      return null;
    }
  }
}

// ── Conversation tile (WhatsApp-style list row) ───────────────────────────────

class _ConvTile extends StatelessWidget {
  final ConversationModel conv;
  final VoidCallback onTap;

  const _ConvTile({required this.conv, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final cs      = Theme.of(context).colorScheme;
    final tt      = Theme.of(context).textTheme;
    final hasNew  = conv.unreadCount > 0;
    final initial = conv.otherUserName.isNotEmpty
        ? conv.otherUserName[0].toUpperCase()
        : '?';

    return InkWell(
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Row(
          children: [
            // Avatar
            CircleAvatar(
              radius: 26,
              backgroundColor:
                  hasNew ? cs.primaryContainer : cs.surfaceContainerHighest,
              child: Text(
                initial,
                style: tt.titleMedium?.copyWith(
                  fontWeight: FontWeight.w700,
                  color: hasNew ? cs.onPrimaryContainer : cs.onSurfaceVariant,
                ),
              ),
            ),
            const SizedBox(width: 12),

            // Name + last message
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    conv.otherUserName,
                    style: tt.bodyMedium?.copyWith(
                      fontWeight:
                          hasNew ? FontWeight.w700 : FontWeight.w600,
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                  const SizedBox(height: 2),
                  Text(
                    conv.lastMessage ?? 'No messages yet',
                    style: tt.bodySmall?.copyWith(
                      color: hasNew ? cs.onSurface : cs.onSurfaceVariant,
                      fontWeight:
                          hasNew ? FontWeight.w600 : FontWeight.normal,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ),
            ),
            const SizedBox(width: 8),

            // Time + unread badge
            Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text(
                  _fmt(conv.lastMessageAt),
                  style: tt.labelSmall?.copyWith(
                    color: hasNew ? cs.primary : cs.onSurfaceVariant,
                    fontWeight:
                        hasNew ? FontWeight.w700 : FontWeight.normal,
                  ),
                ),
                const SizedBox(height: 4),
                if (hasNew)
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 6, vertical: 2),
                    decoration: BoxDecoration(
                      color: cs.primary,
                      borderRadius: Radii.pillRadius,
                    ),
                    child: Text(
                      '${conv.unreadCount}',
                      style: TextStyle(
                        color: cs.onPrimary,
                        fontSize: 11,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  )
                else
                  const SizedBox(height: 18),
              ],
            ),
          ],
        ),
      ),
    );
  }

  String _fmt(DateTime? dt) {
    if (dt == null) return '';
    final now  = DateTime.now();
    final local = dt.toLocal();
    final diff  = now.difference(local);
    if (diff.inMinutes < 1) return 'Now';
    if (diff.inHours   < 1) return '${diff.inMinutes}m';
    if (diff.inDays    < 1) return '${diff.inHours}h';
    if (diff.inDays    < 7) return DateFormat('EEE').format(local);
    return DateFormat('d MMM').format(local);
  }
}

// ── New-chat bottom sheet ─────────────────────────────────────────────────────

class _NewChatSheet extends StatefulWidget {
  final void Function(int convId, String otherName) onStarted;
  const _NewChatSheet({required this.onStarted});

  @override
  State<_NewChatSheet> createState() => _NewChatSheetState();
}

class _NewChatSheetState extends State<_NewChatSheet> {
  int?    _parentUserId;
  String? _parentName;
  final   _msgCtl = TextEditingController();

  @override
  void dispose() {
    _msgCtl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return BlocBuilder<TeacherConversationsCubit, TeacherConversationsState>(
      builder: (context, state) {
        final loaded  = state is TeacherConversationsLoaded ? state : null;
        final parents = loaded?.parents ?? <ParentSummaryModel>[];
        final loading = loaded?.parentsLoading ?? false;
        final starting = loaded?.starting ?? false;

        return SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(20, 4, 20, 24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Parent picker
              DropdownButtonFormField<int?>(
                value: _parentUserId,
                decoration: InputDecoration(
                  labelText: 'Parent *',
                  prefixIcon: loading
                      ? const Padding(
                          padding: EdgeInsets.all(12),
                          child: SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          ),
                        )
                      : const Icon(Icons.person_rounded),
                ),
                hint: loading
                    ? const Text('Loading parents…')
                    : const Text('Select a parent'),
                items: parents
                    .map((p) => DropdownMenuItem<int?>(
                        value: p.userId,
                        child: Text(p.name)))
                    .toList(),
                onChanged: (loading || parents.isEmpty)
                    ? null
                    : (v) {
                        final name = parents
                            .where((p) => p.userId == v)
                            .firstOrNull
                            ?.name ?? '';
                        setState(() {
                          _parentUserId = v;
                          _parentName   = name;
                        });
                      },
              ),
              const SizedBox(height: 14),

              // First message
              TextField(
                controller: _msgCtl,
                minLines: 4,
                maxLines: 8,
                enabled: _parentUserId != null,
                textCapitalization: TextCapitalization.sentences,
                decoration: const InputDecoration(
                  labelText: 'Message *',
                  alignLabelWithHint: true,
                  prefixIcon: Padding(
                    padding: EdgeInsets.only(bottom: 60),
                    child: Icon(Icons.notes_rounded),
                  ),
                ),
              ),
              const SizedBox(height: 20),

              AppButton.primary(
                label: 'Start Chat',
                icon: Icons.send_rounded,
                fullWidth: true,
                size: AppButtonSize.lg,
                loading: starting,
                onPressed: starting ? null : () => _start(context),
              ),
            ],
          ),
        );
      },
    );
  }

  Future<void> _start(BuildContext context) async {
    final body = _msgCtl.text.trim();
    if (_parentUserId == null) {
      _snack(context, 'Please select a parent.');
      return;
    }
    if (body.isEmpty) {
      _snack(context, 'Please write a message.');
      return;
    }
    final convId =
        await context.read<TeacherConversationsCubit>().startConversation(
              parentUserId: _parentUserId!,
              body: body,
            );
    if (!context.mounted) return;
    if (convId != null) {
      widget.onStarted(convId, _parentName ?? 'Parent');
    }
  }

  void _snack(BuildContext context, String msg) =>
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
}
