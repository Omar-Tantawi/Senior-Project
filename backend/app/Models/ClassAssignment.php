<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class ClassAssignment extends Model
{
    protected $table = 'classassignment';
    protected $primaryKey = 'assignment_id';

    protected $fillable = [
        'distribution_id',
        'student_id',
        'class_name',
    ];

    public function student()
    {
        return $this->belongsTo(Student::class, 'student_id', 'id')->with('user');
    }

    public function distribution()
    {
        return $this->belongsTo(ClassDistribution::class, 'distribution_id', 'distribution_id');
    }
}
