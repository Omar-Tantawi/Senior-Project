import 'package:first_try/core/theme/theme.dart';
import 'package:first_try/core/widgets/shared/error_view.dart';
import 'package:first_try/core/widgets/shared/loading_view.dart';
import 'package:first_try/core/widgets/shared/skeletons.dart';
import 'package:first_try/core/widgets/ui/ui.dart';
import 'package:first_try/features/teacher/data/models/teacher_models.dart';
import 'package:first_try/features/teacher/presentation/cubit/teacher_classes_cubit.dart';
import 'package:first_try/features/teacher/presentation/cubit/teacher_classes_state.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

class TeacherClassesScreen extends StatelessWidget {
  const TeacherClassesScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('My Classes',
            style: TextStyle(fontWeight: FontWeight.w700)),
      ),
      body: BlocBuilder<TeacherClassesCubit, TeacherClassesState>(
        builder: (context, state) {
          if (state is TeacherClassesLoading ||
              state is TeacherClassesInitial) {
            return const CardListSkeleton();
          }
          if (state is TeacherClassesError) {
            return ErrorView(
                message: state.message,
                onRetry: () =>
                    context.read<TeacherClassesCubit>().load());
          }
          if (state is! TeacherClassesLoaded) return const SizedBox.shrink();

          final groups = _groupBySection(state.classes);
          return RefreshIndicator(
            onRefresh: () => context.read<TeacherClassesCubit>().load(),
            child: ListView.separated(
              padding: const EdgeInsets.all(16),
              itemCount: groups.length,
              separatorBuilder: (_, __) => const SizedBox(height: 12),
              itemBuilder: (context, i) =>
                  _SectionCard(group: groups[i]),
            ),
          );
        },
      ),
    );
  }
}

// ── Section card ──────────────────────────────────────────────────────────────

class _SectionCard extends StatelessWidget {
  final _SectionGroup group;
  const _SectionCard({required this.group});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final scored = group.students
        .where((s) => s.averageScore != null)
        .toList(growable: false);
    final hasAnyScores = scored.isNotEmpty;
    final avg = hasAnyScores
        ? scored.map((s) => s.averageScore!).reduce((a, b) => a + b) /
            scored.length
        : 0.0;

    return AppCard.surface(
      onTap: () => _showStudents(context, group),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  color: cs.primaryContainer,
                  borderRadius: Radii.smRadius,
                ),
                child:
                    Icon(Icons.groups_rounded, color: cs.onPrimaryContainer),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(group.sectionName,
                        style: const TextStyle(
                            fontWeight: FontWeight.w700, fontSize: 15)),
                    const SizedBox(height: 2),
                    Wrap(
                      spacing: 6,
                      runSpacing: 4,
                      children: group.subjects
                          .map((s) => Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 8, vertical: 2),
                                decoration: BoxDecoration(
                                  color: cs.primaryContainer,
                                  borderRadius: Radii.pillRadius,
                                ),
                                child: Text(s,
                                    style: TextStyle(
                                        fontSize: 11,
                                        fontWeight: FontWeight.w600,
                                        color: cs.onPrimaryContainer)),
                              ))
                          .toList(),
                    ),
                  ],
                ),
              ),
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text('${group.students.length}',
                      style: TextStyle(
                          fontSize: 22,
                          fontWeight: FontWeight.w800,
                          color: cs.primary)),
                  Text('students',
                      style: TextStyle(
                          fontSize: 11, color: cs.onSurfaceVariant)),
                ],
              ),
            ],
          ),
          const SizedBox(height: 12),
          if (hasAnyScores)
            Row(
              children: [
                Text('Class avg:',
                    style:
                        TextStyle(fontSize: 12, color: cs.onSurfaceVariant)),
                const SizedBox(width: 6),
                Expanded(
                  child: ClipRRect(
                    borderRadius: Radii.xsRadius,
                    child: LinearProgressIndicator(
                      value: avg / 100,
                      minHeight: 6,
                      backgroundColor: cs.surfaceContainerHighest,
                      valueColor: AlwaysStoppedAnimation(cs.primary),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Text('${avg.toStringAsFixed(1)}%',
                    style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                        color: cs.primary)),
                if (scored.length < group.students.length) ...[
                  const SizedBox(width: 6),
                  Text('(${scored.length}/${group.students.length})',
                      style: TextStyle(
                          fontSize: 11, color: cs.onSurfaceVariant)),
                ],
              ],
            )
          else
            Row(
              children: [
                Icon(Icons.info_outline_rounded,
                    size: 14, color: cs.onSurfaceVariant),
                const SizedBox(width: 6),
                Text('Not yet graded',
                    style: TextStyle(
                        fontSize: 12,
                        color: cs.onSurfaceVariant,
                        fontStyle: FontStyle.italic)),
              ],
            ),
        ],
      ),
    );
  }

  void _showStudents(BuildContext context, _SectionGroup group) {
    showAppBottomSheet<void>(
      context: context,
      title: group.sectionName,
      subtitle: group.subjects.join(' · '),
      builder: (_) => _StudentsContent(group: group),
    );
  }
}

// ── Students sheet content ────────────────────────────────────────────────────

class _StudentsContent extends StatelessWidget {
  final _SectionGroup group;
  const _StudentsContent({required this.group});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 0, 16, 0),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.only(bottom: 12),
            child: Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color: cs.primaryContainer,
                borderRadius: Radii.pillRadius,
              ),
              child: Text('${group.students.length} students',
                  style: TextStyle(
                      fontSize: 12,
                      color: cs.onPrimaryContainer,
                      fontWeight: FontWeight.w600)),
            ),
          ),
          Flexible(
            child: SingleChildScrollView(
              padding: const EdgeInsets.only(bottom: 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
          ...group.students.map((s) {
            final avg = s.averageScore ?? 0;
            final color = avg >= 85
                ? const Color(0xFF10B981)
                : avg >= 70
                    ? const Color(0xFFF59E0B)
                    : const Color(0xFFE11D48);
            return Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Row(
                children: [
                  CircleAvatar(
                    backgroundColor: cs.primaryContainer,
                    child: Text(s.name[0],
                        style: TextStyle(
                            color: cs.onPrimaryContainer,
                            fontWeight: FontWeight.w700)),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(s.name,
                            style: const TextStyle(
                                fontWeight: FontWeight.w600)),
                        if (s.attendancePercent != null)
                          Text(
                              'Attendance: ${s.attendancePercent!.toStringAsFixed(0)}%',
                              style: TextStyle(
                                  fontSize: 12,
                                  color: cs.onSurfaceVariant)),
                      ],
                    ),
                  ),
                  if (s.averageScore != null)
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: color.withValues(alpha: 0.12),
                        borderRadius: Radii.pillRadius,
                      ),
                      child: Text('${avg.toStringAsFixed(0)}%',
                          style: TextStyle(
                              color: color,
                              fontWeight: FontWeight.w700,
                              fontSize: 13)),
                    ),
                ],
              ),
            );
          }),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ── Grouping ──────────────────────────────────────────────────────────────────

/// A teacher's section, with every subject she teaches it and the merged
/// roster (students deduped by id; per-student average is mean across her
/// subjects for that section; nulls are ignored, not counted as zero).
class _SectionGroup {
  final String sectionName;
  final List<String> subjects;
  final List<ClassStudentModel> students;
  const _SectionGroup({
    required this.sectionName,
    required this.subjects,
    required this.students,
  });
}

List<_SectionGroup> _groupBySection(List<TeacherClassModel> classes) {
  final bySection = <String, List<TeacherClassModel>>{};
  for (final c in classes) {
    bySection.putIfAbsent(c.name, () => []).add(c);
  }

  final groups = bySection.entries.map((entry) {
    final classesForSection = entry.value;
    final subjects = classesForSection.map((c) => c.subject).toSet().toList()
      ..sort();

    // Merge students across this section's subjects, deduping by id.
    final byStudent = <int, List<ClassStudentModel>>{};
    for (final c in classesForSection) {
      for (final s in c.students) {
        byStudent.putIfAbsent(s.id, () => []).add(s);
      }
    }

    final mergedStudents = byStudent.entries.map((e) {
      final variants = e.value;
      // attendance: first non-null we find
      double? attendance;
      for (final v in variants) {
        if (v.attendancePercent != null) {
          attendance = v.attendancePercent;
          break;
        }
      }
      // average score across this teacher's subjects; ignore nulls
      final scoredVariants =
          variants.where((v) => v.averageScore != null).toList();
      final avg = scoredVariants.isEmpty
          ? null
          : scoredVariants
                  .map((v) => v.averageScore!)
                  .reduce((a, b) => a + b) /
              scoredVariants.length;
      return ClassStudentModel(
        id: variants.first.id,
        name: variants.first.name,
        averageScore: avg,
        attendancePercent: attendance,
      );
    }).toList()
      ..sort((a, b) => a.name.compareTo(b.name));

    return _SectionGroup(
      sectionName: entry.key,
      subjects: subjects,
      students: mergedStudents,
    );
  }).toList()
    ..sort((a, b) => a.sectionName.compareTo(b.sectionName));

  return groups;
}
