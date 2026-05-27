import 'package:first_try/core/theme/theme.dart';
import 'package:first_try/core/widgets/shared/empty_state.dart';
import 'package:first_try/core/widgets/shared/error_view.dart';
import 'package:first_try/core/widgets/shared/skeletons.dart';
import 'package:first_try/core/widgets/ui/ui.dart';
import 'package:first_try/features/chat/data/models/chat_models.dart';
import 'package:first_try/features/parent/data/models/parent_extra_models.dart';
import 'package:first_try/features/parent/data/repos/parent_repo.dart';
import 'package:first_try/features/parent/presentation/cubit/parent_chat_cubit.dart';
import 'package:first_try/features/parent/presentation/cubit/parent_conversations_cubit.dart';
import 'package:first_try/features/parent/presentation/screens/parent_chat_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:intl/intl.dart';

class ParentConversationsScreen extends StatefulWidget {
  const ParentConversationsScreen({super.key});

  @override
  State<ParentConversationsScreen> createState() =>
      _ParentConversationsScreenState();
}

class _ParentConversationsScreenState
    extends State<ParentConversationsScreen> {
  @override
  void initState() {
    super.initState();
    context.read<ParentConversationsCubit>().load();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Messages',
            style: TextStyle(fontWeight: FontWeight.w700)),
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () => _showNewChat(context),
        icon: const Icon(Icons.chat_rounded),
        label: const Text('New Chat'),
      ),
      body: BlocConsumer<ParentConversationsCubit, ParentConversationsState>(
        listener: (context, state) {
          if (state is ParentConversationsError) {
            ScaffoldMessenger.of(context).showSnackBar(SnackBar(
              content: Text(state.message),
              backgroundColor: Theme.of(context).colorScheme.error,
            ));
          }
        },
        builder: (context, state) {
          if (state is ParentConversationsInitial ||
              state is ParentConversationsLoading) {
            return const CardListSkeleton(cardHeight: 72);
          }
          if (state is ParentConversationsError) {
            return ErrorView(
              message: state.message,
              onRetry: () => context.read<ParentConversationsCubit>().load(),
            );
          }
          if (state is! ParentConversationsLoaded) {
            return const SizedBox.shrink();
          }

          if (state.conversations.isEmpty) {
            return const EmptyState(
              icon: Icons.chat_bubble_outline_rounded,
              title: 'No conversations yet',
              subtitle: "Tap \"New Chat\" to message one of your child's teachers.",
            );
          }

          return RefreshIndicator(
            onRefresh: () =>
                context.read<ParentConversationsCubit>().refresh(),
            child: ListView.separated(
              padding: const EdgeInsets.fromLTRB(0, 8, 0, 100),
              itemCount: state.conversations.length,
              separatorBuilder: (_, __) =>
                  const Divider(height: 1, indent: 72),
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
    final cubit = context.read<ParentConversationsCubit>();
    final repo  = _getRepo(context);
    if (repo == null) return;

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => BlocProvider(
          create: (_) => ParentChatCubit(
            repo: repo,
            conversationId: conv.id,
          )..load(),
          child: ParentChatScreen(
            conversationId: conv.id,
            otherUserName: conv.otherUserName,
          ),
        ),
      ),
    ).then((_) => cubit.refresh());
  }

  void _showNewChat(BuildContext context) {
    final cubit = context.read<ParentConversationsCubit>();
    cubit.loadTeachers();
    showAppBottomSheet<void>(
      context: context,
      title: 'New Chat',
      subtitle: "Choose a teacher and send a first message.",
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
                  create: (_) => ParentChatCubit(
                    repo: repo,
                    conversationId: convId,
                  )..load(),
                  child: ParentChatScreen(
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

  ParentRepo? _getRepo(BuildContext context) {
    try {
      return context.read<ParentConversationsCubit>().repo;
    } catch (_) {
      return null;
    }
  }
}

// ── Conversation tile ─────────────────────────────────────────────────────────

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
        padding:
            const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Row(
          children: [
            CircleAvatar(
              radius: 26,
              backgroundColor: hasNew
                  ? cs.primaryContainer
                  : cs.surfaceContainerHighest,
              child: Text(
                initial,
                style: tt.titleMedium?.copyWith(
                  fontWeight: FontWeight.w700,
                  color: hasNew
                      ? cs.onPrimaryContainer
                      : cs.onSurfaceVariant,
                ),
              ),
            ),
            const SizedBox(width: 12),
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
                      fontWeight: hasNew
                          ? FontWeight.w600
                          : FontWeight.normal,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ),
            ),
            const SizedBox(width: 8),
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
    final now   = DateTime.now();
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
  int?    _teacherId;
  String? _teacherName;
  final   _msgCtl = TextEditingController();

  @override
  void dispose() {
    _msgCtl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return BlocBuilder<ParentConversationsCubit, ParentConversationsState>(
      builder: (context, state) {
        final loaded   = state is ParentConversationsLoaded ? state : null;
        final teachers = loaded?.teachers ?? <TeacherSummaryModel>[];
        final loading  = loaded?.teachersLoading ?? false;
        final starting = loaded?.starting ?? false;

        return SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(20, 4, 20, 24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              DropdownButtonFormField<int?>(
                value: _teacherId,
                decoration: InputDecoration(
                  labelText: 'Teacher *',
                  prefixIcon: loading
                      ? const Padding(
                          padding: EdgeInsets.all(12),
                          child: SizedBox(
                            width: 20,
                            height: 20,
                            child:
                                CircularProgressIndicator(strokeWidth: 2),
                          ),
                        )
                      : const Icon(Icons.person_rounded),
                ),
                hint: loading
                    ? const Text('Loading teachers…')
                    : const Text('Select a teacher'),
                items: teachers
                    .map((t) => DropdownMenuItem<int?>(
                        value: t.id, child: Text(t.name)))
                    .toList(),
                onChanged: (loading || teachers.isEmpty)
                    ? null
                    : (v) {
                        final name = teachers
                            .where((t) => t.id == v)
                            .firstOrNull
                            ?.name ?? '';
                        setState(() {
                          _teacherId   = v;
                          _teacherName = name;
                        });
                      },
              ),
              const SizedBox(height: 14),

              TextField(
                controller: _msgCtl,
                minLines: 4,
                maxLines: 8,
                enabled: _teacherId != null,
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
    if (_teacherId == null) {
      _snack(context, 'Please select a teacher.');
      return;
    }
    if (body.isEmpty) {
      _snack(context, 'Please write a message.');
      return;
    }
    final convId =
        await context.read<ParentConversationsCubit>().startConversation(
              teacherId: _teacherId!,
              body: body,
            );
    if (!context.mounted) return;
    if (convId != null) {
      widget.onStarted(convId, _teacherName ?? 'Teacher');
    }
  }

  void _snack(BuildContext context, String msg) =>
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
}
